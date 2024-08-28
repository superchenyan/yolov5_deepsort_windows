//
// Created by chentong on 2024/08/23.
//

#include "tracker.h"
#include "Hungarian.h"

ObjectTracker::ObjectTracker(const YAML::Node &config) {
    max_age = config["max_age"].as<int>();
    iou_threshold = config["iou_threshold"].as<float>();
    sim_threshold = config["sim_threshold"].as<float>();
    agnostic = config["agnostic"].as<bool>();
    labels_file = config["labels_file"].as<std::string>();
    class_labels = readClassLabel(labels_file);
    id_colors.resize(100);
    srand((int) time(nullptr));
    for (cv::Scalar &id_color : id_colors)
        id_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

ObjectTracker::~ObjectTracker() = default;

float ObjectTracker::IOUCalculate(const TrackerRes &det_a, const TrackerRes &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}

void ObjectTracker::Alignment(std::vector<std::vector<double>> mat, std::set<int> &unmatchedDetections,
                              std::set<int> &unmatchedTrajectories, std::vector<cv::Point> &matchedPairs,
                              int det_num, int trk_num, bool b_iou) {
    //���洢�������㷨�ķ�����,��СΪtrk_num,Ϊÿ������id�����Ӧ��ƥ����id��δƥ���Ϊ-1
    std::vector<int> assignment;
    HungarianAlgorithm HungAlgo;
    //���洢�������㷨�ķ�����,mat�Ĵ�СΪtrk_num * det_num
    HungAlgo.Solve(mat, trk_num, det_num, assignment);

    //������������,���ڴ洢���п��ܵ���������ƥ�������,���Ͼ���Ψһ��,һ�㲻���ظ�ֵ��
    std::set<int> allItems;
    std::set<int> matchedItems;
    //��ʵ���У��Ƚ�������ƥ�䣬Ȼ�����iouƥ��
    if (b_iou) {

        //��δƥ��ļ��͸��ټ������id��ֵ��
        std::vector<int> detection_index(unmatchedDetections.size());
        std::vector<int> tracker_index(unmatchedTrajectories.size());
        int idx = 0;
        for (const int &umd:unmatchedDetections) {
            detection_index[idx] = umd;
            idx++;
        }
        idx = 0;
        for (const int &umt:unmatchedTrajectories) {
            tracker_index[idx] = umt;
            idx++;
        }
        unmatchedDetections.clear();
        unmatchedTrajectories.clear();
        //�������������ڸ�������������δƥ��ļ�����
        if (det_num > trk_num) { //	there are unmatched detections
            for (unsigned int n = 0; n < det_num; n++)
                allItems.insert(detection_index[n]);

            for (unsigned int i = 0; i < trk_num; ++i)
                matchedItems.insert(detection_index[assignment[i]]);

            //ʹ��set_difference�㷨���ҳ�δƥ��ļ�����, ��δƥ��ļ����������unmatchedDetections
            //����allItems����������matchedItems�ģ�����unmatchedDetections
            set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(),
                           std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (det_num < trk_num) { // there are unmatched trajectory/predictions
            for (unsigned int i = 0; i < trk_num; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(tracker_index[i]);
        }

        //�������й켣�����ݷ�������ƥ����ֵ��ȷ���Ƿ�ƥ��
        for (unsigned int i = 0; i < trk_num; ++i) {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - mat[i][assignment[i]] < iou_threshold) {
                unmatchedTrajectories.insert(tracker_index[i]);
                unmatchedDetections.insert(detection_index[assignment[i]]);
            }
            else
                matchedPairs.emplace_back(cv::Point(tracker_index[i], detection_index[assignment[i]]));
        }
    } else {
        //����ƥ�䣬���������Ŀ���ڸ��ٿ���Ŀ
        if (det_num > trk_num) { //	there are unmatched detections

            for (unsigned int n = 0; n < det_num; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trk_num; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (det_num < trk_num) { // there are unmatched trajectory/predictions��������������det_numΪ0
            for (unsigned int i = 0; i < trk_num; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        }
        //�������й켣�����ݷ�������ƥ����ֵ��ȷ���Ƿ�ƥ�䡣
        for (unsigned int i = 0; i < trk_num; ++i) {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            //���ƥ������ƶȵ��ڽ�������ֵ������Ϊƥ����Ч������Ӧ�Ĺ켣�ͼ�������ӵ�δƥ�伯����
            if (1 - mat[i][assignment[i]] < sim_threshold) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs.emplace_back(cv::Point(i, assignment[i]));
        }
    }
}

void ObjectTracker::FeatureMatching(const std::vector<TrackerRes> &predict_boxes, std::set<int> &unmatchedDetections,
                                    std::set<int> &unmatchedTrajectories, std::vector<cv::Point> &matchedPairs) {
    //��ǰ����Ŀ������
    int det_num = tracker_boxes.size();
    //Ԥ���Ŀ��������
    int trk_num = predict_boxes.size();
    //���trk_num��Ϊ0����det_num��0��������
    if (trk_num == 0 or det_num==0)
        return;
    //����һ�����ƶȾ������ڴ洢ÿ��Ԥ��켣��ÿ��������֮������ƶȷ���
    std::vector<std::vector<double>> similar_mat(trk_num, std::vector<double>(det_num, 0));
    for (unsigned int i = 0; i < trk_num; i++) { // compute iou matrix as a distance matrix
        for (unsigned int j = 0; j < det_num; j++) {
            //���Ԥ��Ĺ켣�ͼ�⵽�Ķ�������ͬһ��𣨻����㷨����Ϊ��𲻿�֪��
            if (predict_boxes[i].classes == tracker_boxes[j].classes or agnostic) {

                // �����������������ĵ����Ȼ����1��ȥ���ֵ����Ϊ���ƶȷ�����
                // ����ʹ��1��ȥ�������Ϊ�������㷨��Ѱ����С�ɱ����䣬��������Ҫ����������ƶ�
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                similar_mat[i][j] = 1 - predict_boxes[i].feature.dot(tracker_boxes[j].feature);
            } else
                similar_mat[i][j] = 1;
        }
    }
    //ȷ��similar_mat��Ϊ��
    Alignment(similar_mat, unmatchedDetections, unmatchedTrajectories, matchedPairs, det_num, trk_num, false);
}

void ObjectTracker::IOUMatching(const std::vector<TrackerRes> &predict_boxes, std::set<int> &unmatchedDetections,
                                std::set<int> &unmatchedTrajectories, std::vector<cv::Point> &matchedPairs) {
    //ֻ������ƥ��ʧ�ܵĽ���iouƥ��
    //��ǰδƥ��ļ���͸��ٿ򼯺�
    int det_num = unmatchedDetections.size();
    int trk_num = unmatchedTrajectories.size();
    if (det_num == 0 or trk_num == 0)
        return;
    std::vector<std::vector<double>> iou_mat(trk_num, std::vector<double>(det_num, 0));
    int i = 0;
    for (const int &umt : unmatchedTrajectories) { // compute iou matrix as a distance matrix
        int j = 0;
        for (const int &umd : unmatchedDetections) {
            if (predict_boxes[umt].classes == tracker_boxes[umd].classes or agnostic) {
                iou_mat[i][j] = 1 - IOUCalculate(predict_boxes[umt], tracker_boxes[umd]);
            } else
                iou_mat[i][j] = 1;
            j++;
        }
        i++;
    }
    Alignment(iou_mat, unmatchedDetections, unmatchedTrajectories, matchedPairs, det_num, trk_num, true);
}

/*
* ÿ֡��Ҫ���ã������㷨�ĺ��Ĳ��֣����������µļ������������֪�Ĺ켣���Լ������µĹ켣��
* ͨ������ƥ��ͽ�����ƥ�䣬�㷨�ܹ�����⵽�Ķ�������֪�Ĺ켣��ƥ�䣬�Ӷ�ʵ�ֶԶ������������
*/
void ObjectTracker::update(const std::vector<DetectRes> &det_boxes, const std::vector<cv::Mat> &det_features, int width, int height) {
    //�����ǰ���ٵı߽���б�
    tracker_boxes.clear();
    int index = 0;
    //���������������ֵ�����ٽ��,idĬ��Ϊ-1
    for (const auto &det_box : det_boxes) {
        TrackerRes tracker_box(det_box.classes, det_box.prob, det_box.x, det_box.y, det_box.w, det_box.h, -1);
        tracker_box.feature = det_features[index];
        index++;
        tracker_boxes.push_back(tracker_box);
    }
    //����������˲����Ĺ켣�б�Ϊ�գ���Ϊÿ����⵽�Ķ��󴴽�һ���µĿ������˲���ʵ����
    if (kalman_boxes.empty()) {
        for (auto &tracker_box : tracker_boxes) {
            StateType rect_box = { tracker_box.x, tracker_box.y, tracker_box.w, tracker_box.h };
            KalmanTracker tracker = KalmanTracker(rect_box, tracker_box.classes, tracker_box.prob);
            tracker.m_feature = tracker_box.feature.clone();
            tracker_box.object_id = tracker.m_id;
            kalman_boxes.push_back(tracker);
        }
        return;
    }
    //����������˲��켣�б�Ϊ��,�������ٽ���б�
    std::vector<TrackerRes> predict_boxes;
    for (auto it = kalman_boxes.begin(); it != kalman_boxes.end();)
    {
        //��ÿ���������˲�������Ԥ�⣬��ȡԤ��ı߽��
        cv::Rect_<float> pBox = (*it).predict();

        //�жϸ��ٿ��Ƿ���Ч���������ٳ���̫��ʱ��max_ageҲ��Ч
        bool is_nan = (isnan(pBox.x)) or (isnan(pBox.y)) or (isnan(pBox.width)) or (isnan(pBox.height));
        bool is_bound = (pBox.x > (float)width) or (pBox.y > (float)height) or
                (pBox.x + pBox.width < 0) or (pBox.y + pBox.height < 0);
        bool is_illegal = (pBox.width <= 0) or (pBox.height <= 0);
        bool time_since_update = it->m_time_since_update > max_age;

        TrackerRes trk_box(it->m_classes, it->m_prob, pBox.x, pBox.y, pBox.width, pBox.height, it->m_id);
        trk_box.classes = it->m_classes;
        trk_box.feature = it->m_feature;
        if (!(time_since_update or is_nan or is_bound or is_illegal))
        {
            //��Ч����������Ԥ����
            predict_boxes.push_back(trk_box);
            it++;
        }
        else
        {   
            //ɾ����ǰ�Ŀ�������
            it = kalman_boxes.erase(it);
            //cerr << "Box invalid at frame: " << frame_count << endl;
        }
    }

    //����һ���������洢δƥ��ļ����������������û���ظ�ֵ
    std::set<int> unmatchedDetections;
    //����һ���������洢δƥ��Ĺ켣������������û���ظ�ֵ
    std::set<int> unmatchedTrajectories;
    //����һ���������洢ƥ��ļ�����͹켣��������
    std::vector<cv::Point> matchedPairs;
    //��������ƥ�亯����ƥ�������͹켣,�Ƚ�������ƥ�䣬Ȼ���ٸ���iouƥ��
    FeatureMatching(predict_boxes, unmatchedDetections, unmatchedTrajectories, matchedPairs);
    //���ý�����ƥ�亯����ƥ�������͹켣
    IOUMatching(predict_boxes, unmatchedDetections, unmatchedTrajectories, matchedPairs);

    //��������ƥ��������ԣ�����ƥ��Ŀ������˲���״̬��
    for (auto & matchedPair : matchedPairs) {
        int trk_id = matchedPair.x;
        int det_id = matchedPair.y;
        StateType rect_box = { tracker_boxes[det_id].x, tracker_boxes[det_id].y,
                               tracker_boxes[det_id].w, tracker_boxes[det_id].h };
        kalman_boxes[trk_id].update(rect_box, tracker_boxes[det_id].classes, tracker_boxes[det_id].prob, tracker_boxes[det_id].feature);
        tracker_boxes[det_id].object_id = kalman_boxes[trk_id].m_id;
    }
    //��������δƥ��ļ�����Ϊ���Ǵ����µĿ������˲���ʵ����
    for (auto umd : unmatchedDetections) {
        StateType rect_box = { tracker_boxes[umd].x, tracker_boxes[umd].y,
                               tracker_boxes[umd].w, tracker_boxes[umd].h };

        //id�ڳ�ʼ��ʱ�򣬽����Զ�����+1
        KalmanTracker tracker = KalmanTracker(rect_box, tracker_boxes[umd].classes, tracker_boxes[umd].prob);
        tracker_boxes[umd].object_id = tracker.m_id;
        tracker.m_feature =  tracker_boxes[umd].feature.clone();
        kalman_boxes.push_back(tracker);
    }
}

void ObjectTracker::DrawResults(cv::Mat &origin_img) {
    cv::cvtColor(origin_img, origin_img, cv::COLOR_BGR2RGB);
    for(const auto &tracker_box : tracker_boxes) {
        char t[256];
        sprintf_s(t, "%d, %f", tracker_box.object_id, tracker_box.prob);
        if (tracker_box.classes != 0) {
            continue;
        }
        std::string name = class_labels[tracker_box.classes] + "-" + t;
        cv::putText(origin_img, name, cv::Point(tracker_box.x - tracker_box.w / 2, tracker_box.y - tracker_box.h / 2 - 5),
                    cv::FONT_HERSHEY_COMPLEX, 0.7, id_colors[tracker_box.object_id % 100], 2);
        cv::Rect rst(tracker_box.x - tracker_box.w / 2, tracker_box.y - tracker_box.h / 2, tracker_box.w, tracker_box.h);
        cv::rectangle(origin_img, rst, id_colors[tracker_box.object_id % 100], 2, cv::LINE_8, 0);
    }
}