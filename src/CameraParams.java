import org.opencv.core.CvType;
import org.opencv.core.Mat;

import Jama.Matrix;

public class CameraParams {

	public int width = 640;
	public int height = 480;

	public float fx = 535.4f;
	public float fy = 539.2f;
	public float cx = 320.1f;
	public float cy = 247.6f;
	public float s = 0;

	public CameraParams() {

	}

	public Matrix getK() {
		Matrix K = Matrix.identity(3, 3);
		K.set(0, 0, fx);
		K.set(0, 1, s);
		K.set(0, 2, cx);
		K.set(1, 1, fy);
		K.set(1, 2, cy);
		return K;
	}

	public Matrix getK4x4() {
		Matrix K = Matrix.identity(4, 4);
		K.set(0, 0, fx);
		K.set(0, 1, s);
		K.set(0, 2, cx);
		K.set(1, 1, fy);
		K.set(1, 2, cy);
		return K;
	}

	public Mat getKMat() {
		Mat K = new Mat(3, 3, CvType.CV_64FC1);
		K.put(0, 0, fx, s, cx, 0, fy, cy, 0, 0, 1);
		return K;
	}

	public float getFx() {
		return fx;
	}

	public float getS() {
		return s;
	}

	public float getCx() {
		return cx;
	}

	public float getFy() {
		return fy;
	}

	public float getCy() {
		return cy;
	}

}
