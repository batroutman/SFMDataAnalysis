import java.util.ArrayList;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

public class Tests {

	public static void testOpticalFlow() {

		String tumFilePath = "../datasets/rgbd_dataset_freiburg3_long_office_household";
		Buffer<FramePack> tumBuf = new TUMBuffer(tumFilePath, true);

		// Create some random colors
		int NUM_FEATURES = 1000;
		Scalar[] colors = new Scalar[NUM_FEATURES];
		Random rng = new Random();
		for (int i = 0; i < NUM_FEATURES; i++) {
			int r = rng.nextInt(256);
			int g = rng.nextInt(256);
			int b = rng.nextInt(256);
			colors[i] = new Scalar(r, g, b);
		}

		Mat old_frame = new Mat(), old_gray = new Mat();
		// Since the function Imgproc.goodFeaturesToTrack requires MatofPoint
		// therefore first p0MatofPoint is passed to the function and then converted to
		// MatOfPoint2f
		MatOfPoint p0MatofPoint = new MatOfPoint();

		FramePack fp0 = tumBuf.getNext();
		old_frame = fp0.getRawFrame();

		Imgproc.cvtColor(old_frame, old_frame, Imgproc.COLOR_RGB2BGR);
		Imgproc.cvtColor(old_frame, old_gray, Imgproc.COLOR_BGR2GRAY);

		Imgproc.goodFeaturesToTrack(old_gray, p0MatofPoint, NUM_FEATURES, 0.01, 15, new Mat(), 7, false, 0.04);
		MatOfPoint2f p0 = new MatOfPoint2f(p0MatofPoint.toArray()), p1 = new MatOfPoint2f();
		Utils.pl("features tracked: " + p0.rows());
		// Create a mask image for drawing purposes
		Mat mask = Mat.zeros(old_frame.size(), old_frame.type());
		while (true) {
			Mat frame = new Mat(), frame_gray = new Mat();

			FramePack fp = tumBuf.getNext();

			if (fp == null) {
				break;
			}

			frame = fp.getRawFrame();

			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2BGR);
			Imgproc.cvtColor(frame, frame_gray, Imgproc.COLOR_BGR2GRAY);
			// calculate optical flow
			MatOfByte status = new MatOfByte();
			MatOfFloat err = new MatOfFloat();
			TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 10, 0.03);
			long start = System.currentTimeMillis();
			Video.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, new Size(30, 30), 2, criteria);
			long end = System.currentTimeMillis();
			Utils.pl("optical flow search time: " + (end - start) + "ms");
			byte StatusArr[] = status.toArray();
			Point p0Arr[] = p0.toArray();
			Point p1Arr[] = p1.toArray();
			ArrayList<Point> good_new = new ArrayList<>();
			ArrayList<Scalar> good_colors = new ArrayList<Scalar>();
			for (int i = 0; i < StatusArr.length; i++) {
				if (StatusArr[i] == 1) {
					good_new.add(p1Arr[i]);
					good_colors.add(colors[i]);
//					Imgproc.line(mask, p1Arr[i], p0Arr[i], colors[i], 2);
					Imgproc.circle(frame, p1Arr[i], 4, colors[i], -1);
				}
			}
			Mat img = new Mat();
			Core.add(frame, mask, img);
			HighGui.imshow("Frame", img);
			int keyboard = HighGui.waitKey(30);
			if (keyboard == 'q' || keyboard == 27) {
				break;
			}
			// Now update the previous frame and previous points
			old_gray = frame_gray.clone();
			Point[] good_new_arr = new Point[good_new.size()];
			good_new_arr = good_new.toArray(good_new_arr);
			p0 = new MatOfPoint2f(good_new_arr);
			good_colors.toArray(colors);
		}
	}

}
