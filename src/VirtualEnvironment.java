import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import Jama.Matrix;

// class to generate 3D points
public class VirtualEnvironment {

	protected CameraParams cameraParams = new CameraParams();
	protected List<Matrix> worldPoints = new ArrayList<Matrix>();
	protected Pose primaryCamera = new Pose();
	protected Pose secondaryCamera = new Pose();

	public VirtualEnvironment() {

	}

	public void generatePoints(int seed, int numPoints, double minX, double maxX, double minY, double maxY, double minZ,
			double maxZ) {

		this.worldPoints.clear();
		Random random = new Random(seed);

		double Z_RANGE = maxZ - minZ;
		double Y_RANGE = maxY - minY;
		double X_RANGE = maxX - minX;

		for (int i = 0; i < numPoints; i++) {
			Matrix point = new Matrix(4, 1);
			point.set(0, 0, random.nextDouble() * X_RANGE + minX);
			point.set(1, 0, random.nextDouble() * Y_RANGE + minY);
			point.set(2, 0, random.nextDouble() * Z_RANGE + minZ);
			point.set(3, 0, 1);
			this.worldPoints.add(point);
		}

	}

	// project image points onto primary camera and return greyscale Mat image
	public Mat getPrimaryImage() {
		return this.getImage(this.primaryCamera, (byte) 0, (byte) 255);
	}

	public Mat getSecondaryImage() {
		return this.getImage(this.secondaryCamera, (byte) 0, (byte) 255);
	}

	public Mat getImage(Pose pose, byte backgroundIntensity, byte foregroundIntensity) {
		byte[] buffer = new byte[this.cameraParams.width * this.cameraParams.height];
		this.fillBuffer(buffer, backgroundIntensity);

		// for each world coordinate, project onto image and add to buffer (if it should
		// be one the image)
		for (int i = 0; i < this.worldPoints.size(); i++) {
			Matrix projCoord = this.cameraParams.getK4x4().times(pose.getHomogeneousMatrix())
					.times(this.worldPoints.get(i));

			// discard if point is behind camera
			if (projCoord.get(2, 0) < 0) {
				continue;
			}
			projCoord = projCoord.times(1 / projCoord.get(2, 0));
			int x = (int) projCoord.get(0, 0);
			int y = (int) projCoord.get(1, 0);

			// add point if it is within camera view
			if (x >= 0 && y >= 0 && x < this.cameraParams.width && y < this.cameraParams.height) {
				buffer[y * this.cameraParams.width + x] = (byte) foregroundIntensity;
			}
		}

		// load buffer into Mat
		Mat mat = new Mat(this.cameraParams.height, this.cameraParams.width, CvType.CV_8U);
		mat.put(0, 0, buffer);
		return mat;
	}

	public void fillBuffer(byte[] buffer, byte fill) {
		for (int i = 0; i < buffer.length; i++) {
			buffer[i] = fill;
		}
	}

	public void savePrimaryImage(String filename) {
		this.saveProjectedImage(this.primaryCamera, filename);
	}

	public void saveSecondaryImage(String filename) {
		this.saveProjectedImage(this.secondaryCamera, filename);
	}

	public void saveProjectedImage(Pose pose, String filename) {
		Mat image = this.getImage(pose, (byte) 0, (byte) 255);
		Imgcodecs.imwrite(filename, image);
	}

	public List<Correspondence2D2D> getCorrespondences() {
		List<Correspondence2D2D> correspondences = new ArrayList<Correspondence2D2D>();

		// for each world point, project into both frames
		// if the point appears in both frames, save a correspondence for it
		for (int i = 0; i < this.worldPoints.size(); i++) {
			Matrix point = this.worldPoints.get(i);

			Matrix projCoordPrimary = this.cameraParams.getK4x4().times(this.primaryCamera.getHomogeneousMatrix())
					.times(point);
			Matrix projCoordSecondary = this.cameraParams.getK4x4().times(this.secondaryCamera.getHomogeneousMatrix())
					.times(point);

			// discard if point is behind camera
			if (projCoordPrimary.get(2, 0) < 0 || projCoordSecondary.get(2, 0) < 0) {
				continue;
			}

			// heterogenize
			projCoordPrimary = projCoordPrimary.times(1 / projCoordPrimary.get(2, 0));
			int x0 = (int) projCoordPrimary.get(0, 0);
			int y0 = (int) projCoordPrimary.get(1, 0);

			projCoordSecondary = projCoordSecondary.times(1 / projCoordSecondary.get(2, 0));
			int x1 = (int) projCoordSecondary.get(0, 0);
			int y1 = (int) projCoordSecondary.get(1, 0);

			// add point if it is within camera view
			if (x0 >= 0 && y0 >= 0 && x0 < this.cameraParams.width && y0 < this.cameraParams.height && x1 >= 0
					&& y1 >= 0 && x1 < this.cameraParams.width && y1 < this.cameraParams.height) {
				Correspondence2D2D c = new Correspondence2D2D(x0, y0, x1, y1);
				correspondences.add(c);
			}
		}

		return correspondences;
	}

	public CameraParams getCameraParams() {
		return cameraParams;
	}

	public void setCameraParams(CameraParams cameraParams) {
		this.cameraParams = cameraParams;
	}

	public List<Matrix> getWorldPoints() {
		return worldPoints;
	}

	public void setWorldPoints(List<Matrix> worldPoints) {
		this.worldPoints = worldPoints;
	}

	public Pose getPrimaryCamera() {
		return primaryCamera;
	}

	public void setPrimaryCamera(Pose primaryCamera) {
		this.primaryCamera = primaryCamera;
	}

	public Pose getSecondaryCamera() {
		return secondaryCamera;
	}

	public void setSecondaryCamera(Pose secondaryCamera) {
		this.secondaryCamera = secondaryCamera;
	}

}