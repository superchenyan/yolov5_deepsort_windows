///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"


int KalmanTracker::kf_count = 0;

// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	//����״̬����ά��Ϊ7
	int stateNum = 7; 
	//�����������ά��Ϊ4
	int measureNum = 4;

	//����һ���������˲�����kf
	kf = cv::KalmanFilter(stateNum, measureNum, 0);
	//��ʼ��һ�����ڴ洢����ֵ�ľ���ά��4*1
	measurement = cv::Mat::zeros(measureNum, 1, CV_32F);
	//��ʼ��״̬ת�ƾ���,7*7
	kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);
	//���ò�������Ϊ��λ��������ζ�Ų�������ֱ�ӵ���״̬������ǰ4��Ԫ�ء�
	setIdentity(kf.measurementMatrix);
	//���ù�������Э�������Ϊ��λ���󣬲�������Ԫ�ص�ֵ����Ϊ0.01�����ʾģ��Ԥ��Ĳ�ȷ���ԡ�
	setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
	//���ò�������Э�������Ϊ��λ���󣬲�������Ԫ�ص�ֵ����Ϊ0.1�����ʾ�����Ĳ�ȷ���ԡ�
	setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
	//���ú������Э�������Ϊ��λ���󣬲�������Ԫ�ص�ֵ����Ϊ1�����ʾ��ʼ״̬���ƵĲ�ȷ���ԡ�
	setIdentity(kf.errorCovPost, cv::Scalar::all(1));
	
	//��ʼ��״̬����
	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
}

// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// ���ÿ������˲�����predict������Ԥ����һʱ�̵�Ŀ���
    cv::Mat p = kf.predict();
	// ���ټ�������
	m_age += 1;

	//�������д���
	if (m_time_since_update > 0)
		m_hit_streak = 0;
	//Ҳ�Ǹ��ټ�����
	m_time_since_update += 1;

	//[cx,cy,s,r]ת��[x,y,w,h]
	StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

	m_history.push_back(predictBox);
	//������ʷ��¼�е����һ��Ԫ�أ�������Ԥ��ı߽��
	return m_history.back();
}

// ������Ƶ֡�й۲쵽�µ�Ŀ��߽��ʱ���������������������˲�����״̬
// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat, int classes, float prob, cv::Mat feature)
{
	//�����ϴθ���������ʱ���������Ϊ0�����ʾĿ���Ѿ����۲쵽������״̬������������
	m_time_since_update = 0;
	//�����ʷ��¼������ܱ�ʾ�ڽ��յ��µĹ۲����ݺ󣬾ɵ�Ԥ����ʷ������Ҫ
	m_history.clear();

	//�������м���������¼��Ŀ�걻�ɹ����ٵĴ���
	m_hits += 1;
	//�����������д����������������¼��Ŀ���������ɹ����ٵĴ���
	m_hit_streak += 1;
	//�洢�۲쵽��Ŀ��������Ϣ
    m_classes = classes;
	//�洢�۲쵽��Ŀ������Ŷ�
    m_prob = prob;
	//��¡���洢�۲쵽��Ŀ�����������,���ں����ʶ���ƥ��
    m_feature = feature.clone();

	// measurement���µĹ۲����������ò�������
	measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	measurement.at<float>(2, 0) = stateMat.area();
	measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

	//����״̬������ʹ�ò�������Ͳ�������Э����������״̬���ƺ����Э����
	kf.correct(measurement);
}


// Return the current state vector
StateType KalmanTracker::get_state()
{
	cv::Mat s = kf.statePost;
	return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}


// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}



/*
// --------------------------------------------------------------------
// Kalman Filter Demonstrating, a 2-d ball demo
// --------------------------------------------------------------------

const int winHeight = 600;
const int winWidth = 800;

Point mousePosition = Point(winWidth >> 1, winHeight >> 1);

// mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_MOUSEMOVE) {
		mousePosition = Point(x, y);
	}
}

void TestKF();

void main()
{
	TestKF();
}


void TestKF()
{
	int stateNum = 4;
	int measureNum = 2;
	KalmanFilter kf = KalmanFilter(stateNum, measureNum, 0);

	// initialization
	Mat processNoise(stateNum, 1, CV_32F);
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = *(Mat_<float>(stateNum, stateNum) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kf.errorCovPost, Scalar::all(1));

	randn(kf.statePost, Scalar::all(0), Scalar::all(winHeight));

	namedWindow("Kalman");
	setMouseCallback("Kalman", mouseEvent);
	Mat img(winHeight, winWidth, CV_8UC3);

	while (1)
	{
		// predict
		Mat prediction = kf.predict();
		Point predictPt = Point(prediction.at<float>(0, 0), prediction.at<float>(1, 0));

		// generate measurement
		Point statePt = mousePosition;
		measurement.at<float>(0, 0) = statePt.x;
		measurement.at<float>(1, 0) = statePt.y;

		// update
		kf.correct(measurement);

		// visualization
		img.setTo(Scalar(255, 255, 255));
		circle(img, predictPt, 8, CV_RGB(0, 255, 0), -1); // predicted point as green
		circle(img, statePt, 8, CV_RGB(255, 0, 0), -1); // current position as red

		imshow("Kalman", img);
		char code = (char)waitKey(100);
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}
	destroyWindow("Kalman");
}
*/
