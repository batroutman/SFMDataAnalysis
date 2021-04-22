import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import Jama.Matrix;

// class to generate 3D points
public class VirtualEnvironment {

	public static double PROJECTION_NOISE_VARIANCE = 9;

	protected CameraParams cameraParams = new CameraParams();
	protected List<Matrix> worldPoints = new ArrayList<Matrix>();
	protected Pose primaryCamera = new Pose();
	protected Pose secondaryCamera = new Pose();

	public VirtualEnvironment() {

	}

	public void generatePoints(int seed, int numPoints, double minX, double maxX, double minY, double maxY, double minZ,
			double maxZ) {

		this.worldPoints.clear();

//		this.worldPoints.addAll(this.getPointsInPlane(seed, numPoints, 0, 0.1, 1, 0, 1, 1, 2, 2, 2, 2));
//		this.worldPoints.addAll(this.getPointsInSphere(seed, 1000, 0, 0, 0, 0.5, 0.5));
		this.worldPoints.addAll(this.getPointsInPlane(seed, numPoints, 0, 0, 1, 0, 0, 1, 2, 2, 2, 0));

	}

	public void generatePlanarScene(long seed, int numPoints) {
		this.worldPoints.clear();

		this.worldPoints.addAll(this.getPointsInPlane(seed, numPoints, 0, 0, 1, 0, 0, 1, 2, 2, 2, 0.03)); // 0.03
	}

	public void generateSphericalScene(long seed, int numPoints) {
		this.worldPoints.clear();

		this.worldPoints.addAll(this.getPointsInSphere(seed, numPoints, 0, 0, 0, 1, 1));
	}

	public void generateScene0(long seed) {
		this.worldPoints.clear();

		// desk front
		this.worldPoints.addAll(this.getPointsInPlane(seed, 40, 0, 1, 3, 0, 0, 1, 3, 1, 0, 0));

		// desk backboard
		this.worldPoints.addAll(this.getPointsInPlane(seed, 40, 0, 0, 4, 0, 0, 1, 3, 1, 0, 0));

		// desk backboard picture
		this.worldPoints.addAll(this.getPointsInPlane(seed, 50, 0.5, 0, 4, 0, 0, 1, 0.5, 0.5, 0, 0));

		// desk top
		this.worldPoints.addAll(this.getPointsInPlane(seed, 20, 0, 0.5, 3.5, 0, -1, 0, 3, 1, 1, 0));
	}

	public List<Matrix> getPointsInPlane(long seed, int numPoints, double x0, double y0, double z0, double normalX,
			double normalY, double normalZ, double xRange, double yRange, double zRange, double noiseRange) {

		List<Matrix> points = new ArrayList<Matrix>();

		// calculate minimum values for each dimension
		double xMin = x0 - xRange / 2;
		double yMin = y0 - yRange / 2;
		double zMin = z0 - zRange / 2;

		Random rand = new Random(seed);

		// generate points
		for (int i = 0; i < numPoints; i++) {

			double x = rand.nextDouble() * xRange + xMin;
			double y = rand.nextDouble() * yRange + yMin;
			double z = rand.nextDouble() * zRange + zMin;

			// correct one of the coordinates
			if (normalZ != 0) {
				z = (-normalX * (x - x0) - normalY * (y - y0)) / normalZ + z0;
			} else if (normalX != 0) {
				x = (-normalZ * (z - z0) - normalY * (y - y0)) / normalX + x0;
			} else if (normalY != 0) {
				y = (-normalZ * (z - z0) - normalX * (x - x0)) / normalY + y0;
			}

			double noiseX = rand.nextDouble() * noiseRange - noiseRange / 2;
			double noiseY = rand.nextDouble() * noiseRange - noiseRange / 2;
			double noiseZ = rand.nextDouble() * noiseRange - noiseRange / 2;

			x += noiseX;
			y += noiseY;
			z += noiseZ;

			Matrix p = new Matrix(4, 1);
			p.set(0, 0, x);
			p.set(1, 0, y);
			p.set(2, 0, z);
			p.set(3, 0, 1);
			points.add(p);

		}

		return points;

	}

	public List<Matrix> getPointsInSphere(long seed, int numPoints, double x0, double y0, double z0, double radius,
			double noiseRange) {

		List<Matrix> points = new ArrayList<Matrix>();

		Random rand = new Random(seed);

		for (int i = 0; i < numPoints; i++) {
			double theta = rand.nextDouble() * 2 * Math.PI;
			double phi = rand.nextDouble() * Math.PI;

			double x = radius * Math.cos(theta) * Math.sin(phi) + x0;
			double y = radius * Math.sin(theta) * Math.sin(phi) + y0;
			double z = radius * Math.cos(phi) + z0;

			double noiseX = rand.nextDouble() * noiseRange - noiseRange / 2;
			double noiseY = rand.nextDouble() * noiseRange - noiseRange / 2;
			double noiseZ = rand.nextDouble() * noiseRange - noiseRange / 2;

			x += noiseX;
			y += noiseY;
			z += noiseZ;

			Matrix p = new Matrix(4, 1);
			p.set(0, 0, x);
			p.set(1, 0, y);
			p.set(2, 0, z);
			p.set(3, 0, 1);
			points.add(p);
		}

		return points;

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

		Random gaussRand = new Random(1777);

		// for each world coordinate, project onto image and add to buffer (if it should
		// be one the image)
		for (int i = 0; i < this.worldPoints.size(); i++) {
			Matrix projCoord = this.getProjection(pose, this.worldPoints.get(i), gaussRand, PROJECTION_NOISE_VARIANCE);

			// discard if point is behind camera
			if (projCoord == null) {
				continue;
			}

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
		return this.getCorrespondences(new ArrayList<Matrix>());
	}

	public List<Correspondence2D2D> getCorrespondences(List<Matrix> oTruePoints) {

		// output of the world coordinates that were on the screen (same order as
		// correspondences)
		oTruePoints.clear();

		List<Correspondence2D2D> correspondences = new ArrayList<Correspondence2D2D>();
		Random gaussRand = new Random(1777);

		// for each world point, project into both frames
		// if the point appears in both frames, save a correspondence for it
		for (int i = 0; i < this.worldPoints.size(); i++) {
			Matrix point = this.worldPoints.get(i);

			Matrix projCoordPrimary = this.getProjection(this.primaryCamera, point, gaussRand,
					PROJECTION_NOISE_VARIANCE);
			Matrix projCoordSecondary = this.getProjection(this.secondaryCamera, point, gaussRand,
					PROJECTION_NOISE_VARIANCE);

			// discard if point is behind camera
			if (projCoordPrimary == null || projCoordSecondary == null) {
				continue;
			}

			// convert to ints
			int x0 = (int) projCoordPrimary.get(0, 0);
			int y0 = (int) projCoordPrimary.get(1, 0);

			int x1 = (int) projCoordSecondary.get(0, 0);
			int y1 = (int) projCoordSecondary.get(1, 0);

			// add point if it is within camera view
			if (x0 >= 0 && y0 >= 0 && x0 < this.cameraParams.width && y0 < this.cameraParams.height && x1 >= 0
					&& y1 >= 0 && x1 < this.cameraParams.width && y1 < this.cameraParams.height) {
				Correspondence2D2D c = new Correspondence2D2D(x0, y0, x1, y1);
				correspondences.add(c);
				oTruePoints.add(point);
			}
		}

		return correspondences;
	}

	// given a camera and a 4D homogeneous point, return the homogeneous projection
	// of that point onto the camera.
	// if projected point is behind camera, return null
	public Matrix getProjection(Pose pose, Matrix point, Random rand, double noiseVariance) {

		// get projection
		Matrix projCoord = this.cameraParams.getK4x4().times(pose.getHomogeneousMatrix()).times(point);

		// test if in front of camera
		if (projCoord.get(2, 0) <= 0) {
			return null;
		}

		// homogenize projection
		projCoord = projCoord.times(1 / projCoord.get(2, 0));

		// add gaussian noise to point
		double xNoise = rand.nextGaussian() * Math.sqrt(noiseVariance);
		double yNoise = rand.nextGaussian() * Math.sqrt(noiseVariance);

		projCoord.set(0, 0, projCoord.get(0, 0) + xNoise);
		projCoord.set(1, 0, projCoord.get(1, 0) + yNoise);

		return projCoord;

	}

	public Matrix getTrueFundamentalMatrix() {
		Pose pose = this.secondaryCamera;
		Matrix K = this.cameraParams.getK();

		Matrix R = pose.getRotationMatrix().getMatrix(0, 2, 0, 2);
		Matrix tx = new Matrix(3, 3);
		tx.set(0, 1, -pose.getTz());
		tx.set(0, 2, pose.getTy());
		tx.set(1, 0, pose.getTz());
		tx.set(1, 2, -pose.getTx());
		tx.set(2, 0, -pose.getTy());
		tx.set(2, 1, pose.getTx());

		Matrix essential = tx.times(R);

		Matrix KInv = K.getMatrix(0, 2, 0, 2).inverse();

		Matrix fundamental = KInv.transpose().times(essential).times(KInv);

		return fundamental;
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

	public Pose getSecondaryCamera() {
		return secondaryCamera;
	}

	public void setSecondaryCamera(Pose secondaryCamera) {
		this.secondaryCamera = secondaryCamera;
	}

}
