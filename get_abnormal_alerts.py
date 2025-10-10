import os
import requests
from datetime import datetime, timedelta
import requests
import logging
from zoneinfo import ZoneInfo

# KSA timezone
ksa_tz = ZoneInfo("Asia/Riyadh")

alerts_date = (datetime.now(ksa_tz) - timedelta(days=1)).strftime('%Y-%m-%d')

LOG_PATH = os.path.join("/data1/yasir/Data/", str(alerts_date))
os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE = LOG_PATH + "abnormal.log"

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

def get_auth_token():

    # API endpoint
    url = "https://dsw-bk.core9ventures.com/api/Users/authenticate"
    log.info(f"Authenticate URL: {url}")

    # Payload
    payload = {
        "username": "AI_MODEL_S",
        "password": "MOdel@25"  
    }

    log.info(f"Authenticate Payload: {payload}")

    try:
        # Make the POST request
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()

            token = data.get('token')

            return token
        else:
            log.error(f"API request failed. Status code: {response.status_code}")
            log.error(f"Response: {response.text}")

            return None
        
    except Exception as e:
        log.error(f"Exception during authentication: {e}")

def get_alerts_api(auth_token):

    # API endpoint
    url = "https://drivesensebk.core9.ai/api/Alert/GetAlertDataAnalysisV2"
    log.info(f"URL: {url}")

    # Headers
    headers = {
        "Authorization": f"{auth_token}",
        "Content-Type": "application/json"
    }

    # Get previous day's date
    base_date = datetime.now(ksa_tz) - timedelta(days=1)

    # Specify start and end time for payload
    start_time = base_date.replace(hour=0, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")
    end_time = base_date.replace(hour=23, minute=59, second=59).strftime("%Y-%m-%d %H:%M:%S")

    # Payload
    payload = {
        "alarmType": [
            "driverAnomaly"
        ],
        "startTime": start_time,
        "endTime": end_time,
        "orderBy": "ASC"
    }

    #print("Payload: ", payload)
    log.info(f"Payload: {payload}")

    # Base URL to prepend to filePath
    IMAGE_BASE_URL = "http://87.237.226.169:20003/"

    # Directory to save downloaded images
    SAVE_DIR = "/data1/yasir/Data/Abnormal Alerts"
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)

        # Handle the response
        if response.status_code == 200:
            data = response.json()

            alerts = data.get("result")

            log.info(f"Total alerts found: {len(alerts)}")

            # === LOG START TIME ===
            alerts_start_time = datetime.now()
            log.info(f"Alerts download started at: {alerts_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            total_alerts = str(len(alerts))
            log.info(f"Downloading {total_alerts} alerts...")

            downloaded_alerts = 0
            skipped_alerts = 0

            for alert in alerts:
                #alarm_no = alert.get("alarmNo")
                alarm_no = str(alert.get("vaId"))
                file_paths = alert.get("filePaths", [])
                alert_time = alert.get("startTime")  # "2025-04-13 10:00:09"
                process_status = alert.get("processStatus")
                license_num = alert.get("licenseNum")

                log.info(f"Downloading Alert: {alarm_no}")

                at = alert_time

                if not alarm_no or not alert_time:
                    skipped_alerts += 1
                    log.info(f"Alert has no vaId or startTime value: {alarm_no}")
                    continue

                if process_status == 1:
                    skipped_alerts += 1
                    log.info(f"Alert {alarm_no} has already been processed. Skipping download.")
                    continue

                # Format: "YYYY-MM-DD HH:MM:SS" -> "YYYYMMDD_HHMMSS"
                dt = datetime.strptime(alert_time, "%Y-%m-%d %H:%M:%S")
                date_str = dt.strftime("%Y-%m-%d")
                timestamp_str = dt.strftime("%Y%m%d_%H%M%S")

                # Folder path: seatbelt/YYYY-MM-DD/alarmNo/
                folder_path = os.path.join(SAVE_DIR, date_str, alarm_no)
                os.makedirs(folder_path, exist_ok=True)

                info_file = os.path.join(folder_path, "info.txt")

                # Write info to the file
                with open(info_file, "w") as f:
                    f.write(f"license_num={license_num}\n")
                    f.write(f"alert_date_time={at}\n")

                if not file_paths:
                    log.info(f"Missing attachment for alert: {alarm_no}")
                    missing_file = os.path.join(folder_path, "missing.txt")

                    with open(missing_file, "w") as f:
                        pass  # creates an empty file

                for idx, file_path in enumerate(file_paths, start=1):
                    ext = os.path.splitext(file_path)[-1].lower()  # Get extension

                    if ext == ".jpg":
                        #print("skipping image: ", alarm_no)
                        continue

                    suffix = "_video" if ext == ".mp4" else f"_{idx}" # Suffix for video

                    new_filename = f"{alarm_no}_{timestamp_str}{suffix}{ext}"
                    full_url = IMAGE_BASE_URL + file_path
                    save_path = os.path.join(folder_path, new_filename)

                    if os.path.exists(save_path):
                        continue           

                    # Download the image
                    try:
                        img_response = requests.get(full_url)
                        if img_response.status_code == 200:
                            with open(save_path, 'wb') as f:
                                f.write(img_response.content)
                            #print(f"Image saved to: {save_path}\n")

                        else:
                            log.error(f"Failed to download file. Status code: {img_response.status_code}")
                    except Exception as e:
                        log.error(f"Error downloading file: {e}")

                downloaded_alerts += 1
            
            log.info(f"Total alerts skipped: {skipped_alerts}") 
            log.info(f"Total alerts downloaded: {downloaded_alerts}")
            
            # === LOG END TIME ===
            alerts_end_time = datetime.now()
            duration = alerts_end_time - alerts_start_time
            log.info(f"Alerts download completed at: {alerts_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            log.info(f"Duration: {str(duration)}")

        else:
            log.error(f"API request failed. Status code: {response.status_code}")
            log.error(f"Response: {response.text}")

    except Exception as e:
        log.error(f"Exception during getting alerts: {e}")

if __name__ == "__main__":
    auth_token = get_auth_token()
    if auth_token is not None:
        get_alerts_api(auth_token)