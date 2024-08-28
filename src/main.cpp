/*
 * @Author: chentong
 * @Date: 2024-08-23 04:14:53
 * @LastEditTime: 2024-08-23 12:57:29
 * @LastEditors: chentong
 * @Description: 
 * @FilePath: /yolov5_deepsort_tensorrt/src/main.cpp
 * 
 */
#include <iostream>
#include "yaml-cpp/yaml.h"
#include "video.h"

#include<stdio.h>
#include<string.h>
//#include<unistd.h>

int main() {
    
    std::string config_file = "E:\\code\\yolov5\\yolov5\\yolov5\\configs\\config_v5.yaml";
    std::string video_name = "E:\\code\\yolov5\\yolov5\\yolov5\\dataset\\001.avi";
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node detect_config = root["detect"];
    YAML::Node tracker_config = root["tracker"];
    YAML::Node fastreid_config = root["fastreid"];
    YOLO detect(detect_config);
    detect.LoadEngine();
    ObjectTracker tracker(tracker_config);
    fastreid fastreid(fastreid_config);
    fastreid.LoadEngine();
    //InferenceVideo(video_name, detect, tracker, fastreid);
    InferenceCamere(detect, tracker, fastreid);
    return  0;
}