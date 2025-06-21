import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from enum import Enum
import queue
import shutil
from functools import partial

import cv2
import requests
import runpod
import yt_dlp
# Removed gofile import
import service.trans_dh_service
from h_utils.custom import CustomError
from y_utils.config import GlobalConfig
from y_utils.logger import logger

# Initialize GlobalConfig (assuming it's necessary)
GlobalConfig.load_config("config/config.ini")

def download_file(url, save_path):
    """Downloads a file from a URL."""
    if "youtube.com/" in url or "youtu.be/" in url:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': save_path,
            'merge_output_format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return save_path
    else:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path

def upload_to_gofile(file_path):
    """Uploads a file to gofile.io and returns the download link."""
    try:
        g = Gofile()
        response = g.upload(file=file_path)
        # Ensure the command is formatted correctly
        command = [
            "curl", "--upload-file", file_path,
            f"https://transfer.sh/{os.path.basename(file_path)}"
        ]
        logger.info(f"Executing transfer.sh upload: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        download_url = process.stdout.strip()
        if not download_url.startswith("http"):
            logger.error(f"transfer.sh upload failed. Output: {download_url}")
            return f"Failed to upload {os.path.basename(file_path)}. Details: {download_url}"
        return download_url
    except subprocess.CalledProcessError as e:
        logger.error(f"transfer.sh upload failed with CalledProcessError: {e}. Output: {e.stderr}")
        return f"Failed to upload {os.path.basename(file_path)}. Error: {e.stderr}"
    except FileNotFoundError:
        logger.error("curl command not found. Please ensure curl is installed in the environment.")
        return "Failed to upload due to missing curl."
    except Exception as e:
        logger.error(f"An unexpected error occurred during transfer.sh upload: {e}")
        return f"Failed to upload {os.path.basename(file_path)}. Unexpected error: {str(e)}"

def write_video_runpod(
    output_imgs_queue,
    temp_dir,
    result_dir,
    work_id,
    audio_path,
    result_queue,
    width,
    height,
    fps,
    watermark_switch=0,
    digital_auth=0,
    temp_queue=None, # Added temp_queue to match expected signature if needed
):
    output_mp4 = os.path.join(temp_dir, "{}-t.mp4".format(work_id))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result_path = os.path.join(result_dir, "{}-r.mp4".format(work_id)) # Ensure result_dir exists
    os.makedirs(result_dir, exist_ok=True) # Create result_dir if it doesn't exist
    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    print("Custom VideoWriter init done")
    try:
        while True:
            state, reason, value_ = output_imgs_queue.get(timeout=600) # Added timeout
            if type(state) == bool and state == True:
                logger.info(
                    "Custom VideoWriter [{}]视频帧队列处理已结束".format(work_id)
                )
                logger.info(
                    "Custom VideoWriter Silence Video saved in {}".format(
                        os.path.realpath(output_mp4)
                    )
                )
                video_write.release()
                break
            else:
                if type(state) == bool and state == False:
                    logger.error(
                        "Custom VideoWriter [{}]任务视频帧队列 -> 异常原因:[{}]".format(
                            work_id, reason
                        )
                    )
                    raise CustomError(reason)
                for result_img in value_:
                    video_write.write(result_img)
        if video_write.isOpened(): # Check if video_write is still open
            video_write.release()

        # FFMPEG command construction (similar to app.py)
        # Ensure GlobalConfig paths are correctly loaded or set defaults
        watermark_path = getattr(GlobalConfig.instance(), 'watermark_path', None)
        digital_auth_path = getattr(GlobalConfig.instance(), 'digital_auth_path', None)

        if watermark_switch == 1 and digital_auth == 1 and watermark_path and digital_auth_path:
            logger.info(
                "Custom VideoWriter [{}]任务需要水印和数字人标识".format(work_id)
            )
            # Simplified command for brevity, actual command might need adjustment
            command = f'ffmpeg -y -i {audio_path} -i {output_mp4} -i {watermark_path} -i {digital_auth_path} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {result_path}'
        elif watermark_switch == 1 and watermark_path:
            logger.info("Custom VideoWriter [{}]任务需要水印".format(work_id))
            command = f'ffmpeg -y -i {audio_path} -i {output_mp4} -i {watermark_path} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10" -c:a aac -crf 15 -strict -2 {result_path}'
        elif digital_auth == 1 and digital_auth_path:
            logger.info("Custom VideoWriter [{}]任务需要数字人标识".format(work_id))
            command = f'ffmpeg -loglevel warning -y -i {audio_path} -i {output_mp4} -i {digital_auth_path} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {result_path}'
        else:
            command = f"ffmpeg -loglevel warning -y -i {audio_path} -i {output_mp4} -c:a aac -c:v libx264 -crf 15 -strict -2 {result_path}"

        logger.info("Custom command:{}".format(command))
        subprocess.call(command, shell=True)
        print("###### Custom Video Writer write over")
        print(f"###### Video result saved in {os.path.realpath(result_path)}")
        result_queue.put([True, result_path])
    except queue.Empty:
        logger.error(f"Custom VideoWriter [{work_id}] timed out waiting for image queue.")
        result_queue.put([False, f"[{work_id}] Timed out waiting for image queue."])
        if video_write.isOpened():
            video_write.release()
    except Exception as e:
        logger.error(
            "Custom VideoWriter [{}]视频帧队列处理异常结束，异常原因:[{}]".format(
                work_id, e.__str__()
            )
        )
        result_queue.put(
            [
                False,
                "[{}]视频帧队列处理异常结束，异常原因:[{}]".format(
                    work_id, e.__str__()
                ),
            ]
        )
        if video_write.isOpened(): # Ensure release on exception
            video_write.release()
    logger.info("Custom VideoWriter 后处理进程结束")


# Monkey patch the write_video function in the service module
service.trans_dh_service.write_video = write_video_runpod


class VideoProcessor:
    def __init__(self):
        self.task = service.trans_dh_service.TransDhTask()
        self.basedir = GlobalConfig.instance().result_dir
        self.is_initialized = False
        self._initialize_service()
        print("VideoProcessor init done")

    def _initialize_service(self):
        logger.info("开始初始化 trans_dh_service...")
        try:
            # Simulating initialization time, replace with actual check if possible
            # In a serverless environment, long initializations should be handled carefully
            # or done during container startup if feasible.
            time.sleep(10) # Increased sleep to ensure service is ready
            logger.info("trans_dh_service 初始化完成。")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"初始化 trans_dh_service 失败: {e}")
            # This should ideally raise an error or signal unreadiness
            self.is_initialized = False


    def process_video_internal(
        self, audio_file_path, video_file_path, watermark=False, digital_auth=False
    ):
        if not self.is_initialized:
            logger.error("Service not initialized. Cannot process video.")
            # In a real scenario, might try to re-initialize or raise a specific error
            self._initialize_service() # Attempt to initialize again
            if not self.is_initialized:
                 return {"error": "Service failed to initialize"}


        work_id = str(uuid.uuid1())
        temp_dir = os.path.join(GlobalConfig.instance().temp_dir, work_id)
        os.makedirs(temp_dir, exist_ok=True)
        # result_dir should be specific to this request to avoid clashes
        # and allow cleanup.
        # Using a subfolder of GlobalConfig.instance().result_dir for this run
        current_result_dir = os.path.join(GlobalConfig.instance().result_dir, work_id)
        os.makedirs(current_result_dir, exist_ok=True)

        try:
            cap = cv2.VideoCapture(video_file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # The core task execution
            # Ensure `self.task.work` uses the correct `write_video` function
            # and paths are correctly managed.
            # The `code` argument seems to be the work_id.
            # The other integer arguments (0,0,0,0) might be flags or placeholders.
            # Their exact meaning would require deeper inspection of `TransDhTask.work`.
            self.task.task_dic[work_id] = "" # Initialize task entry
            self.task.work(audio_file_path, video_file_path, work_id, 0, 0, 0, 0)


            # Assuming self.task.task_dic[work_id] will be populated with results
            # The structure of task_dic[code] was [status, message, result_path] in app.py
            # We need to wait for the result_queue which is now handled by write_video_runpod

            # Create a queue to get the result path from write_video_runpod
            result_processing_queue = queue.Queue()

            # The result of task.work is now implicitly handled by how write_video_runpod
            # is called by the underlying service. We need to ensure that the
            # result_path is correctly retrieved.
            # The original app.py retrieves result_path from self.task.task_dic[code][2]
            # This implies task.work populates this. Let's assume it still does,
            # and then ffmpeg processing happens.

            # We need to ensure the result_path from the ffmpeg processing in write_video_runpod
            # is correctly obtained. The current `write_video_runpod` puts it in `result_queue`.
            # `TransDhTask` needs to be aware of this queue or we need a way to pass it.

            # Let's assume `task.work` internally uses the patched `write_video`
            # which uses `result_queue` passed to it.
            # We need to pass this queue to `task.work` if possible, or find how `task.work`
            # communicates the final processed video path.

            # Re-evaluating: `write_video_gradio` (now `write_video_runpod`) is called by
            # `TransDhTask`. `TransDhTask` itself creates and manages `output_imgs_queue`
            # and `result_queue` which are passed to `write_video`.
            # So, `self.task.work` should eventually cause `result_queue.put([True, result_path])`
            # to be called within `write_video_runpod`.

            # The `task.task_dic[code]` was used in app.py to get the *intermediate* result path
            # before ffmpeg. We need the *final* path after ffmpeg.
            # The `TransDhTask` object itself has `result_queue_map` which stores the result queue
            # for each work_id.

            if work_id in self.task.result_queue_map:
                final_result_queue = self.task.result_queue_map[work_id]
                status, final_result_path_or_msg = final_result_queue.get(timeout=600) # Wait for ffmpeg result
                if status:
                    # Move final result to a predictable location if not already
                    # The result_path from write_video_runpod is already in current_result_dir
                    final_video_path = final_result_path_or_msg
                    logger.info(f"Processing successful. Output video: {final_video_path}")

                    uploaded_url = upload_to_gofile(final_video_path)
                    return {"output_video_url": uploaded_url}
                else:
                    logger.error(f"Video processing failed: {final_result_path_or_msg}")
                    return {"error": f"Video processing failed: {final_result_path_or_msg}"}
            else:
                logger.error(f"Result queue not found for work_id: {work_id}")
                return {"error": f"Result queue not found for work_id: {work_id}. This indicates an internal issue."}

        except CustomError as e:
            logger.error(f"CustomError during video processing: {e}")
            return {"error": str(e)}
        except queue.Empty:
            logger.error(f"Timeout waiting for result from processing task for work_id: {work_id}")
            return {"error": "Processing timed out"}
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            return {"error": f"An unexpected error occurred: {str(e)}"}
        finally:
            # Cleanup temporary files and directories
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            # Optionally, clean up the specific result_dir if files are uploaded elsewhere
            # For now, keeping it for inspection if needed, or RunPod handles workspace cleanup.
            # if os.path.exists(current_result_dir):
            #     shutil.rmtree(current_result_dir)
            if work_id in self.task.task_dic:
                del self.task.task_dic[work_id]
            if work_id in self.task.result_queue_map:
                del self.task.result_queue_map[work_id]
            if work_id in self.task.output_imgs_queue_map:
                del self.task.output_imgs_queue_map[work_id]
            gc.collect()


# Global instance of VideoProcessor
# This is important for serverless environments to reuse the initialized model
# if the container is warm.
video_processor_instance = None

def ensure_processor_initialized():
    global video_processor_instance
    if video_processor_instance is None or not video_processor_instance.is_initialized:
        logger.info("Video processor not initialized or in bad state. Initializing...")
        video_processor_instance = VideoProcessor()
        if not video_processor_instance.is_initialized:
            # This is a critical failure if the processor cannot be initialized.
            # Subsequent calls will likely fail.
            logger.error("CRITICAL: Video processor failed to initialize.")
    return video_processor_instance


def handler(event):
    """
    RunPod Serverless Handler
    """
    global video_processor_instance

    # Ensure processor is initialized
    # This might take time on the first call (cold start)
    try:
        current_processor = ensure_processor_initialized()
        if not current_processor.is_initialized:
             return {"error": "Video processor could not be initialized."}
    except Exception as e:
        logger.error(f"Failed to initialize VideoProcessor: {e}")
        return {"error": f"Failed to initialize VideoProcessor: {str(e)}"}


    job_input = event.get("input", {})
    video_url = job_input.get("video_url")
    audio_url = job_input.get("audio_url")
    # Optional: watermark and digital_auth flags
    watermark = job_input.get("watermark", False)
    digital_auth = job_input.get("digital_auth", False)


    if not video_url or not audio_url:
        return {"error": "Missing video_url or audio_url in input"}

    # Create temporary directories for downloaded files
    # Unique download directory for each request
    request_id = str(uuid.uuid4())
    download_dir = os.path.join(GlobalConfig.instance().temp_dir, "downloads", request_id)
    os.makedirs(download_dir, exist_ok=True)

    downloaded_video_path = os.path.join(download_dir, "video.mp4")
    downloaded_audio_path = os.path.join(download_dir, "audio.wav") # Assuming wav, adjust if needed

    try:
        logger.info(f"Downloading video from: {video_url}")
        download_file(video_url, downloaded_video_path)
        logger.info(f"Video downloaded to: {downloaded_video_path}")

        logger.info(f"Downloading audio from: {audio_url}")
        download_file(audio_url, downloaded_audio_path)
        logger.info(f"Audio downloaded to: {downloaded_audio_path}")

        # Process the video using the downloaded files
        result = current_processor.process_video_internal(
            downloaded_audio_path,
            downloaded_video_path,
            watermark=watermark,
            digital_auth=digital_auth
        )
        return result

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error: {e}")
        return {"error": f"Failed to download video/audio: {str(e)}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Requests download error: {e}")
        return {"error": f"Failed to download video/audio: {str(e)}"}
    except Exception as e:
        logger.error(f"Error in handler: {e}")
        traceback.print_exc()
        return {"error": f"An unexpected error occurred in handler: {str(e)}"}
    finally:
        # Clean up downloaded files
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
        gc.collect()


if __name__ == "__main__":
    # This part is for local testing of the handler if needed,
    # but RunPod will call the handler function directly.
    logger.info("Starting RunPod serverless worker...")

    # Ensure necessary directories exist before starting the server
    # These paths should ideally be configurable or use standard temp locations
    # Ensure GlobalConfig is initialized if not already
    if not hasattr(GlobalConfig, '_instance'):
        GlobalConfig.load_config("config/config.ini") # Or your default config path

    temp_dir_path = GlobalConfig.instance().temp_dir
    result_dir_path = GlobalConfig.instance().result_dir

    if not os.path.exists(temp_dir_path):
        os.makedirs(temp_dir_path)
        logger.info(f"Created temp directory: {temp_dir_path}")
    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)
        logger.info(f"Created result directory: {result_dir_path}")

    runpod.serverless.start({"handler": handler})
