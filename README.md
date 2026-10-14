
# NVIDIA DeepStream Basic Application

## 1. Project Objective

The objective of this project is to build a basic NVIDIA DeepStream application for video processing and object detection.

The project includes:
- Processing a single video source.
- Configuring multiple video sources.
- Running primary inference using a pretrained TrafficCamNet model.
- Combining multiple source frames using the DeepStream tiled-display configuration.
- Saving processed video output.
- Documenting setup, configuration, execution, and results.

## 2. Development Environment

- Development laptop: Windows
- Execution environment: Remote Ubuntu GPU server
- NVIDIA DeepStream: 7.0
- Input format: MP4
- Model: ResNet18 TrafficCamNet
- Model format: NVIDIA TAO ETLT
- Version control: Git and GitHub

## 3. Project Structure

```text
Deepstream_Basic_application/
├── config/
│   ├── config_infer_primary.txt
│   ├── deepstream_app.txt
│   └── deepstream_multi_source.txt
├── input/
│   ├── sample.mp4
│   ├── sample2.mp4
│   ├── sample3.mp4
│   └── sample4.mp4
├── models/
│   └── Primary_Detector/
│       ├── resnet18_trafficcamnet.etlt
│       ├── labels.txt
│       └── cal_trt.bin
├── output/
│   ├── deepstream_output.mp4
│   └── multi_source_output.mp4
├── logs/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

Note: `multi_source_output.mp4` is the intended multi-source output file. It will be present only after a successful multi-source run.

## 4. Input Videos

The `input/` directory contains four MP4 files.

- `sample.mp4`: Original cricket video.
- `sample2.mp4`: Copy of the original video.
- `sample3.mp4`: Copy of the original video.
- `sample4.mp4`: Copy of the original video.

These files are intended to act as four separate DeepStream sources. The footage is identical, but each file is configured as a separate source.

## 5. Model Files

The `models/Primary_Detector/` directory contains the pretrained model and supporting files.

- `resnet18_trafficcamnet.etlt`: Pretrained TrafficCamNet model.
- `labels.txt`: Class labels used by the model.
- `cal_trt.bin`: INT8 calibration data.

TrafficCamNet is designed for traffic-scene object detection. Its detection quality on cricket footage may be limited.

## 6. Configuration Files

### 6.1 config_infer_primary.txt

This file configures the primary inference model.

It specifies model-related settings such as:
- Model file path.
- TensorRT engine path.
- Label file path.
- Calibration file path.
- Batch size.
- Network precision and input dimensions.

### 6.2 deepstream_app.txt

This is the original single-source application configuration.

It specifies:
- Input video path.
- Stream multiplexer settings.
- Primary inference component.
- On-screen display settings.
- Output video sink.

The output is configured as:

`output/deepstream_output.mp4`

### 6.3 deepstream_multi_source.txt

This is the new configuration prepared for four video sources.

It specifies:
- Four enabled video sources.
- A stream multiplexer batch size of four.
- Primary inference batch size of four.
- A 2-row by 2-column tiled-display layout.
- A separate output file path.

The intended output is:

`output/multi_source_output.mp4`

The multi-source configuration must be tested on the Ubuntu GPU server before it can be considered complete.

## 7. DeepStream Pipeline Components

### Video Sources

Each source reads one local MP4 file. Four source sections are configured to process the four sample videos.

### Stream Multiplexer

The stream multiplexer combines frames from multiple sources into batches for downstream processing.

The multi-source configuration uses a batch size of four.

### Primary Inference

The primary inference component uses the TrafficCamNet model to perform object detection on the video frames.

### Tiled Display

The tiled-display component arranges frames from multiple sources in a grid.

The multi-source configuration uses two rows and two columns.

### On-Screen Display

The OSD component draws supported detection information and labels on frames.

### Output Sink

The file output sink encodes and writes the processed video to an MP4 file.

## 8. Running the Single-Source Application

Run these commands on the Ubuntu GPU server from the project directory:

```bash
cd ~/Documents/Interns/Indra/Deepstream_Basic_application
deepstream-app -c config/deepstream_app.txt
```

The application previously completed successfully and reported:

- TensorRT engine generated successfully.
- Model loaded.
- End-of-stream event received.
- `App run successful`.

The generated output was:

`output/deepstream_output.mp4`

The output file was reported to be approximately 3.6 MB.

## 9. Preparing the Multi-Source Application

The multi-source configuration has been created on Windows.

Before running it on Ubuntu:

1. Ensure all four sample videos are available.
2. Ensure the model and configuration files are present.
3. Check that the Ubuntu paths match the actual project location.
4. Confirm the configured inference engine path and batch-size settings.
5. Ensure the output directory exists.
6. Run the application and inspect the output.

## 10. Running the Multi-Source Application

On the Ubuntu GPU server, run:

```bash
cd ~/Documents/Interns/Indra/Deepstream_Basic_application
deepstream-app -c config/deepstream_multi_source.txt
```

The intended output file is:

`output/multi_source_output.mp4`

Expected checks:
- All four sources open successfully.
- The inference component loads successfully.
- The tiled output contains frames from all four sources.
- The application reaches the end of the streams without errors.
- The output video is created and can be played.

These are expected results, not yet confirmed results. Update this section after testing.

## 11. TensorRT Engine

DeepStream can generate a TensorRT engine from the configured model when an appropriate engine is not available.

The generated engine is excluded from Git because it can be environment-specific. Keep the original model, labels, calibration data, and inference configuration in the repository.

## 12. Git and GitHub

Repository:

https://github.com/indra112284/Deepstream_Basic_application

The original single-source configuration, model files, and generated single-source output were pushed to the `main` branch.

After the multi-source configuration has been tested and the documentation updated, commit and push the new changes.

Typical commands:

```bash
git status
git add README.md config/deepstream_multi_source.txt input/
git commit -m "Add multi-source DeepStream configuration and documentation"
git push origin main
```

Add the multi-source output file only after it has been generated and verified.

## 13. Troubleshooting

### Input video not found

Check the video path and ensure the file exists on the Ubuntu server.

### Model or engine loading error

Check the paths in both inference configuration files and confirm that the model files are present.

### Batch-size mismatch

Check that the number of configured sources and the relevant batch-size settings are appropriate for the pipeline.

### Output video not created

Check the output path, write permissions, encoder availability, and terminal logs.

### Multi-source pipeline fails

Check the DeepStream logs and verify each source independently before investigating the full pipeline.

## 14. Current Project Status

### Completed

- Single-source DeepStream application executed successfully.
- TrafficCamNet model loaded.
- TensorRT engine generated.
- Single-source MP4 output generated.
- Model files and configuration committed to GitHub.
- Four sample video files are present in the Windows project.
- Multi-source configuration file prepared on Windows.
- README documentation updated.

### Pending Verification

- Run the multi-source configuration on Ubuntu.
- Confirm all four sources are processed.
- Verify the tiled output and detection results.
- Confirm the multi-source output video is generated.
- Record the actual multi-source execution results.
- Push the final tested configuration and documentation to GitHub.
