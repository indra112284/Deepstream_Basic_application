
import argparse
import os
import sys


def load_gstreamer():
    """Load GStreamer Python bindings from the target environment."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GLib

        return Gst, GLib

    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "GStreamer Python bindings are unavailable. "
            "Run this application in the Ubuntu DeepStream environment "
            "with compatible GStreamer and PyGObject installed."
        ) from exc


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Basic NVIDIA DeepStream video detection application"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an input MP4 video",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the primary inference configuration",
    )
    parser.add_argument(
        "--output",
        default="output/detected_video.mp4",
        help="Path to the output MP4 video",
    )
    return parser.parse_args()


def validate_paths(args):
    input_path = os.path.abspath(args.input)
    config_path = os.path.abspath(args.config)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Inference configuration not found: {config_path}"
        )

    if os.path.getsize(config_path) == 0:
        raise ValueError(
            f"Inference configuration is empty: {config_path}. "
            "Configure the selected detection model before running."
        )

    if input_path == output_path:
        raise ValueError(
            "Input and output paths must be different."
        )

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    return input_path, config_path, output_path


def make_element(Gst, factory_name, element_name):
    element = Gst.ElementFactory.make(
        factory_name, element_name
    )

    if element is None:
        raise RuntimeError(
            f"Could not create GStreamer element '{factory_name}'. "
            "Check the DeepStream installation and available plugins."
        )

    return element


def request_mux_sink_pad(streammux, pad_name):
    """Request a streammux sink pad across common GStreamer APIs."""
    if hasattr(streammux, "request_pad_simple"):
        return streammux.request_pad_simple(pad_name)

    # Compatibility fallback for older GStreamer releases.
    return streammux.get_request_pad(pad_name)


def build_pipeline(Gst, input_path, config_path, output_path):
    pipeline = Gst.Pipeline.new("deepstream-basic-pipeline")

    if pipeline is None:
        raise RuntimeError(
            "Could not create the GStreamer pipeline."
        )

    source = make_element(Gst, "filesrc", "video-source")
    demux = make_element(Gst, "qtdemux", "mp4-demux")
    decoder = make_element(
        Gst, "nvv4l2decoder", "video-decoder"
    )
    streammux = make_element(
        Gst, "nvstreammux", "stream-muxer"
    )
    inference = make_element(
        Gst, "nvinfer", "primary-inference"
    )
    converter = make_element(
        Gst, "nvvideoconvert", "video-converter"
    )
    osd = make_element(
        Gst, "nvdsosd", "onscreen-display"
    )
    output_converter = make_element(
        Gst, "nvvideoconvert", "output-converter"
    )
    encoder = make_element(
        Gst, "nvv4l2h264enc", "video-encoder"
    )
    h264_parser = make_element(
        Gst, "h264parse", "h264-parser"
    )
    mp4_muxer = make_element(
        Gst, "qtmux", "mp4-muxer"
    )
    sink = make_element(
        Gst, "filesink", "output-file"
    )

    elements = [
        source,
        demux,
        decoder,
        streammux,
        inference,
        converter,
        osd,
        output_converter,
        encoder,
        h264_parser,
        mp4_muxer,
        sink,
    ]

    for element in elements:
        pipeline.add(element)

    source.set_property("location", input_path)
    sink.set_property("location", output_path)
    sink.set_property("sync", False)

    # Basic single-video streammux settings.
    # Verify property support on the target DeepStream version.
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("batched-push-timeout", 40000)

    inference.set_property(
        "config-file-path", config_path
    )

    if not source.link(demux):
        raise RuntimeError(
            "Could not link filesrc to qtdemux."
        )

    # qtdemux creates its video pad dynamically.
    # Connect that video pad to the decoder sink.
    link_state = {"video_linked": False, "error": None}

    def on_demux_pad_added(_demux, pad):
        caps = pad.get_current_caps()

        if caps is None:
            caps = pad.query_caps(None)

        if caps is None or caps.get_size() == 0:
            return

        media_type = caps.get_structure(0).get_name()

        # Ignore audio and other non-video streams.
        if not media_type.startswith("video/"):
            return

        decoder_sink = decoder.get_static_pad("sink")

        if decoder_sink is None:
            link_state["error"] = (
                "Could not get decoder sink pad."
            )
            return

        if decoder_sink.is_linked():
            return

        result = pad.link(decoder_sink)

        if result == Gst.PadLinkReturn.OK:
            link_state["video_linked"] = True
            print("Linked MP4 video stream to decoder.")
        else:
            link_state["error"] = (
                f"Demux-to-decoder link failed: {result}"
            )
            print(link_state["error"])

    demux.connect("pad-added", on_demux_pad_added)

    # The decoder source pad is static. Request the mux sink pad
    # and link them before starting the pipeline.
    decoder_src = decoder.get_static_pad("src")

    if decoder_src is None:
        raise RuntimeError(
            "Could not get the decoder source pad."
        )

    mux_sink = request_mux_sink_pad(
        streammux, "sink_0"
    )

    if mux_sink is None:
        raise RuntimeError(
            "Could not request nvstreammux sink_0."
        )

    result = decoder_src.link(mux_sink)

    if result != Gst.PadLinkReturn.OK:
        raise RuntimeError(
            f"Decoder-to-streammux link failed: {result}"
        )

    links = [
        (streammux, inference),
        (inference, converter),
        (converter, osd),
        (osd, output_converter),
        (output_converter, encoder),
        (encoder, h264_parser),
        (h264_parser, mp4_muxer),
        (mp4_muxer, sink),
    ]

    for upstream, downstream in links:
        if not upstream.link(downstream):
            raise RuntimeError(
                f"Could not link {upstream.get_name()} "
                f"to {downstream.get_name()}."
            )

    return pipeline, link_state


def run_pipeline(Gst, GLib, pipeline, link_state):
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    state = {"success": False, "error": None}

    def on_message(_bus, message):
        if message.type == Gst.MessageType.EOS:
            print("End of stream: processing completed.")
            state["success"] = True
            loop.quit()

        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            state["error"] = str(error)

            print(f"GStreamer error: {error}")

            if debug:
                print(f"Debug details: {debug}")

            loop.quit()

    bus.connect("message", on_message)

    print("Starting DeepStream pipeline...")

    result = pipeline.set_state(Gst.State.PLAYING)

    if result == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        raise RuntimeError(
            "Pipeline failed to enter PLAYING state."
        )

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()

    if link_state["error"]:
        raise RuntimeError(link_state["error"])

    if not link_state["video_linked"]:
        raise RuntimeError(
            "No video stream was linked from qtdemux to the decoder. "
            "Check that the input file contains a supported video stream."
        )

    if state["error"]:
        raise RuntimeError(
            f"Pipeline processing failed: {state['error']}"
        )

    if not state["success"]:
        raise RuntimeError(
            "Pipeline did not complete successfully."
        )


def main():
    args = parse_arguments()

    input_path, config_path, output_path = validate_paths(args)

    Gst, GLib = load_gstreamer()
    Gst.init(None)

    pipeline, link_state = build_pipeline(
        Gst,
        input_path,
        config_path,
        output_path,
    )

    print(f"Input:  {input_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")

    run_pipeline(
        Gst,
        GLib,
        pipeline,
        link_state,
    )

    if not os.path.isfile(output_path):
        raise RuntimeError(
            "Processing finished, but the output file was not created."
        )

    if os.path.getsize(output_path) == 0:
        raise RuntimeError(
            "The output video is empty."
        )

    print(
        f"Success. Output saved to: {output_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
