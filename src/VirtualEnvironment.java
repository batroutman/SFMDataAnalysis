import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

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

	public Matrix getTrueFundamentalMatrix() {
		Matrix Pprime = this.cameraParams.getK4x4().times(this.secondaryCamera.getHomogeneousMatrix()).getMatrix(0, 2,
				0, 3);
		Matrix P = this.cameraParams.getK4x4().times(this.primaryCamera.getHomogeneousMatrix()).getMatrix(0, 2, 0, 3);

		// get pseudo-inverse of P
		SingularValueDecomposition svd = P.transpose().svd();
		Matrix sigmaPlus = svd.getS().inverse();
		Matrix PtPlus = svd.getV().times(sigmaPlus).times(svd.getU().transpose());
		Matrix Pplus = PtPlus.transpose();

		// get epipole
		Matrix camCenter = new Matrix(4, 1);
		camCenter.set(0, 0, this.primaryCamera.getCx());
		camCenter.set(1, 0, this.primaryCamera.getCy());
		camCenter.set(2, 0, this.primaryCamera.getCz());
		camCenter.set(3, 0, 1);
		Matrix epipole = Pprime.times(camCenter); // homogenize?

		Matrix et = new Matrix(3, 3);
		et.set(0, 1, -epipole.get(2, 0));
		et.set(0, 2, epipole.get(1, 0));
		et.set(1, 0, epipole.get(2, 0));

		et.set(1, 2, -epipole.get(0, 0));
		et.set(2, 0, -epipole.get(1, 0));
		et.set(2, 1, epipole.get(0, 0));

		Matrix F = et.times(Pprime).times(Pplus);
		F.print(15, 5);

		return F;
	}

	public Matrix estimateFundamentalMatrix(List<Correspondence2D2D> correspondences) {

		// create point matrices
		List<Point> points0 = new ArrayList<Point>();
		List<Point> points1 = new ArrayList<Point>();
		for (int i = 0; i < correspondences.size(); i++) {
			Point point0 = new Point(correspondences.get(i).getX0(), correspondences.get(i).getY0());
			Point point1 = new Point(correspondences.get(i).getX1(), correspondences.get(i).getY1());
			points0.add(point0);
			points1.add(point1);
		}

		MatOfPoint2f points0Mat = new MatOfPoint2f();
		MatOfPoint2f points1Mat = new MatOfPoint2f();
		points0Mat.fromList(points0);
		points1Mat.fromList(points1);

		long start = System.currentTimeMillis();
		Mat fundamentalMatrix = Calib3d.findFundamentalMat(points0Mat, points1Mat, Calib3d.FM_7POINT);
//		Mat fundamentalMatrix = Calib3d.findFundamentalMat(points0Mat, points1Mat, Calib3d.FM_RANSAC, 2, 0.99, 500);
		long end = System.currentTimeMillis();
		Utils.pl("Fundamental matrix estimation time: " + (end - start) + "ms");

		return Utils.MatToMatrix(fundamentalMatrix);
	}

	public Matrix getPoseFromFundamentalMatrix(Matrix fundamentalMatrix, List<Correspondence2D2D> correspondences) {

		Mat funMat = Utils.MatrixToMat(fundamentalMatrix);

		// convert to essential matrix
		Mat K = this.cameraParams.getKMat();
		Mat Kt = new Mat();
		Core.transpose(K, Kt);
		Mat E = new Mat();
		Core.gemm(Kt, funMat, 1, new Mat(), 0, E, 0);
		Core.gemm(E, K, 1, new Mat(), 0, E, 0);

		// decompose essential matrix
		Mat R1Mat = new Mat();
		Mat R2Mat = new Mat();
		Mat tMat = new Mat();
		Calib3d.decomposeEssentialMat(E, R1Mat, R2Mat, tMat);

		Matrix R1 = Utils.MatToMatrix(R1Mat);
		Matrix R2 = Utils.MatToMatrix(R2Mat);
		Matrix t = Utils.MatToMatrix(tMat);

		// triangulate point and select correct solution (chirality)
		Matrix I = Matrix.identity(4, 4);
		Matrix R1t1 = Matrix.identity(4, 4);
		Matrix R1t2 = Matrix.identity(4, 4);
		Matrix R2t1 = Matrix.identity(4, 4);
		Matrix R2t2 = Matrix.identity(4, 4);
		Matrix[] possiblePoses = { R1t1, R1t2, R2t1, R2t2 };

		R1t1.setMatrix(0, 2, 0, 2, R1);
		R1t1.setMatrix(0, 2, 3, 3, t);
		R1t2.setMatrix(0, 2, 0, 2, R1);
		R1t2.setMatrix(0, 2, 3, 3, t.times(-1));
		R2t1.setMatrix(0, 2, 0, 2, R2);
		R2t1.setMatrix(0, 2, 3, 3, t);
		R2t2.setMatrix(0, 2, 0, 2, R2);
		R2t2.setMatrix(0, 2, 3, 3, t.times(-1));

		Random rand = new Random();
		int[] scores = { 0, 0, 0, 0 };

		for (int i = 0; i < 32 && i < correspondences.size(); i++) {
			int index = (int) (rand.nextDouble() * correspondences.size());
			Correspondence2D2D c = correspondences.get(index);

			// get triangulated 3D points
			Matrix point3DR1t1 = triangulate(R1t1, I, c);
			Matrix point3DR1t2 = triangulate(R1t2, I, c);
			Matrix point3DR2t1 = triangulate(R2t1, I, c);
			Matrix point3DR2t2 = triangulate(R2t2, I, c);

			// reproject points onto cameras
			Matrix point2DR1t1 = R1t1.times(point3DR1t1);
			Matrix point2DR1t2 = R1t2.times(point3DR1t2);
			Matrix point2DR2t1 = R2t1.times(point3DR2t1);
			Matrix point2DR2t2 = R2t2.times(point3DR2t2);

			int numSelected = 0;
			if (point3DR1t1.get(2, 0) > 0 && point2DR1t1.get(2, 0) > 0) {
				scores[0]++;
				numSelected++;
			}
			if (point3DR1t2.get(2, 0) > 0 && point2DR1t2.get(2, 0) > 0) {
				scores[1]++;
				numSelected++;
			}
			if (point3DR2t1.get(2, 0) > 0 && point2DR2t1.get(2, 0) > 0) {
				scores[2]++;
				numSelected++;
			}
			if (point3DR2t2.get(2, 0) > 0 && point2DR2t2.get(2, 0) > 0) {
				scores[3]++;
				numSelected++;
			}
			if (numSelected > 1) {
				Utils.pl(
						"UH OH! More than one pose passed acceptance criteria in fundamental matrix initialization! (Photogrammetry::SFMFundamentalMatrixEstimate()) ==> numSelected: "
								+ numSelected);
			}

		}

		// find highest scoring pose
		int highestInd = 0;
		for (int i = 1; i < scores.length; i++) {
			highestInd = scores[i] > scores[highestInd] ? i : highestInd;
		}

		// convert to quaternion and pose object
		Matrix selection = possiblePoses[highestInd];

		return selection;

	}

	public Matrix triangulate(Matrix E, Matrix pose, Correspondence2D2D c) {

		Matrix Pprime = E.times(pose);
		Pprime = this.cameraParams.getK4x4().times(Pprime);

		Matrix P = this.cameraParams.getK4x4().times(pose);

		// compute A matrix for Ax = 0
		Matrix row0 = P.getMatrix(2, 2, 0, 3).times(c.getX0()).minus(P.getMatrix(0, 0, 0, 3));
		Matrix row1 = P.getMatrix(2, 2, 0, 3).times(c.getY0()).minus(P.getMatrix(1, 1, 0, 3));
		Matrix row2 = Pprime.getMatrix(2, 2, 0, 3).times(c.getX1()).minus(Pprime.getMatrix(0, 0, 0, 3));
		Matrix row3 = Pprime.getMatrix(2, 2, 0, 3).times(c.getY1()).minus(Pprime.getMatrix(1, 1, 0, 3));

		Matrix A = new Matrix(4, 4);
		A.setMatrix(0, 0, 0, 3, row0);
		A.setMatrix(1, 1, 0, 3, row1);
		A.setMatrix(2, 2, 0, 3, row2);
		A.setMatrix(3, 3, 0, 3, row3);

		SingularValueDecomposition svd = A.svd();
		Matrix X = svd.getV().getMatrix(0, 3, 3, 3);
		X = X.times(1.0 / X.get(3, 0));
		// System.out.println("X");
		// X.print(5, 4);
		return X;
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
