///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"


int KalmanTracker::kf_count = 0;

// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	//定义状态向量维度为7
	int stateNum = 7; 
	//定义测量向量维度为4
	int measureNum = 4;

	//创建一个卡尔曼滤波对象kf
	kf = cv::KalmanFilter(stateNum, measureNum, 0);
	//初始化一个用于存储测量值的矩阵，维度4*1
	measurement = cv::Mat::zeros(measureNum, 1, CV_32F);
	//初始化状态转移矩阵,7*7
	kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);
	//设置测量矩阵为单位矩阵。这意味着测量向量直接等于状态向量的前4个元素。
	setIdentity(kf.measurementMatrix);
	//设置过程噪声协方差矩阵为单位矩阵，并将所有元素的值设置为0.01。这表示模型预测的不确定性。
	setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
	//设置测量噪声协方差矩阵为单位矩阵，并将所有元素的值设置为0.1。这表示测量的不确定性。
	setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
	//设置后验误差协方差矩阵为单位矩阵，并将所有元素的值设置为1。这表示初始状态估计的不确定性。
	setIdentity(kf.errorCovPost, cv::Scalar::all(1));
	
	//初始化状态向量
	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
}

// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// 调用卡尔曼滤波器的predict方法来预测下一时刻的目标框
    cv::Mat p = kf.predict();
	// 跟踪计数器，
	m_age += 1;

	//连续命中次数
	if (m_time_since_update > 0)
		m_hit_streak = 0;
	//也是跟踪计数器
	m_time_since_update += 1;

	//[cx,cy,s,r]转成[x,y,w,h]
	StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

	m_history.push_back(predictBox);
	//返回历史记录中的最后一个元素，即最新预测的边界框
	return m_history.back();
}

// 当从视频帧中观察到新的目标边界框时，会调用这个函数来更新滤波器的状态
// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat, int classes, float prob, cv::Mat feature)
{
	//将自上次更新以来的时间计数重置为0。这表示目标已经被观察到，并且状态向量将被更新
	m_time_since_update = 0;
	//清除历史记录。这可能表示在接收到新的观察数据后，旧的预测历史不再需要
	m_history.clear();

	//增加命中计数器，记录了目标被成功跟踪的次数
	m_hits += 1;
	//增加连续命中次数。这个计数器记录了目标连续被成功跟踪的次数
	m_hit_streak += 1;
	//存储观察到的目标的类别信息
    m_classes = classes;
	//存储观察到的目标的置信度
    m_prob = prob;
	//克隆并存储观察到的目标的特征向量,用于后面的识别和匹配
    m_feature = feature.clone();

	// measurement，新的观察量用来设置测量矩阵
	measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	measurement.at<float>(2, 0) = stateMat.area();
	measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

	//更新状态向量。使用测量矩阵和测量噪声协方差来更新状态估计和误差协方差
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
