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

		this.worldPoints.addAll(this.getPointsInPlane(seed, numPoints, 0, 0.1, 1, 0, 1, 1, 2, 2, 2, 0));
//		this.worldPoints.addAll(this.getPointsInSphere(seed, 1000, 0, 0, 0, 0.5, 0.5));

//		Random random = new Random(seed);
//
//		double Z_RANGE = maxZ - minZ;
//		double Y_RANGE = maxY - minY;
//		double X_RANGE = maxX - minX;
//
//		for (int i = 0; i < numPoints; i++) {
//			Matrix point = new Matrix(4, 1);
//			point.set(0, 0, random.nextDouble() * X_RANGE + minX);
//			point.set(1, 0, random.nextDouble() * Y_RANGE + minY);
//			point.set(2, 0, random.nextDouble() * Z_RANGE + minZ);
//			point.set(3, 0, 1);
//			this.worldPoints.add(point);
//		}

	}

	public List<Matrix> getPointsInPlane(int seed, int numPoints, double x0, double y0, double z0, double normalX,
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

	public List<Matrix> getPointsInSphere(int seed, int numPoints, double x0, double y0, double z0, double radius,
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

//	public Matrix getTrueHomography() {
//
//	}

	public Mat estimateHomography(List<Correspondence2D2D> correspondences) {

		ArrayList<Point> matchedKeyframePoints = new ArrayList<Point>();
		ArrayList<Point> matchedPoints = new ArrayList<Point>();

		for (int i = 0; i < correspondences.size(); i++) {
			Correspondence2D2D c = correspondences.get(i);
			Point point1 = new Point();
			Point point2 = new Point();
			point1.x = c.getX0();
			point1.y = c.getY0();
			point2.x = c.getX1();
			point2.y = c.getY1();
			matchedKeyframePoints.add(point1);
			matchedPoints.add(point2);
		}

		// compute homography
		MatOfPoint2f keyframeMat = new MatOfPoint2f();
		MatOfPoint2f matKeypoints = new MatOfPoint2f();
		keyframeMat.fromList(matchedKeyframePoints);
		matKeypoints.fromList(matchedPoints);
		Mat homography = Calib3d.findHomography(keyframeMat, matKeypoints);

		return homography;

	}

	public Matrix getPoseFromHomography(Mat homography, List<Correspondence2D2D> correspondences) {

		Mat intrinsics = this.cameraParams.getKMat();
		List<Mat> rotations = new ArrayList<Mat>();
		List<Mat> translations = new ArrayList<Mat>();
		List<Mat> normals = new ArrayList<Mat>();
		Calib3d.decomposeHomographyMat(homography, intrinsics, rotations, translations, normals);

		Matrix E = this.selectHomographySolution(rotations, translations, correspondences);
		return E;
	}

	public Matrix selectHomographySolution(List<Mat> rotations, List<Mat> translations,
			List<Correspondence2D2D> correspondences) {

		Matrix pose = this.primaryCamera.getHomogeneousMatrix();

		Matrix R1 = Utils.MatToMatrix(rotations.get(0));
		Matrix R2 = Utils.MatToMatrix(rotations.get(1));
		Matrix R3 = Utils.MatToMatrix(rotations.get(2));
		Matrix R4 = Utils.MatToMatrix(rotations.get(3));

		Matrix t1 = Utils.MatToMatrix(translations.get(0));
		Matrix t2 = Utils.MatToMatrix(translations.get(1));
		Matrix t3 = Utils.MatToMatrix(translations.get(2));
		Matrix t4 = Utils.MatToMatrix(translations.get(3));

		// set up extrinsic matrices (all possible options)
		Matrix E1 = Matrix.identity(4, 4);
		Matrix E2 = Matrix.identity(4, 4);
		Matrix E3 = Matrix.identity(4, 4);
		Matrix E4 = Matrix.identity(4, 4);

		E1.set(0, 0, R1.get(0, 0));
		E1.set(0, 1, R1.get(0, 1));
		E1.set(0, 2, R1.get(0, 2));
		E1.set(1, 0, R1.get(1, 0));
		E1.set(1, 1, R1.get(1, 1));
		E1.set(1, 2, R1.get(1, 2));
		E1.set(2, 0, R1.get(2, 0));
		E1.set(2, 1, R1.get(2, 1));
		E1.set(2, 2, R1.get(2, 2));
		E1.set(0, 3, t1.get(0, 0));
		E1.set(1, 3, t1.get(1, 0));
		E1.set(2, 3, t1.get(2, 0));

		E2.set(0, 0, R2.get(0, 0));
		E2.set(0, 1, R2.get(0, 1));
		E2.set(0, 2, R2.get(0, 2));
		E2.set(1, 0, R2.get(1, 0));
		E2.set(1, 1, R2.get(1, 1));
		E2.set(1, 2, R2.get(1, 2));
		E2.set(2, 0, R2.get(2, 0));
		E2.set(2, 1, R2.get(2, 1));
		E2.set(2, 2, R2.get(2, 2));
		E2.set(0, 3, t2.get(0, 0));
		E2.set(1, 3, t2.get(1, 0));
		E2.set(2, 3, t2.get(2, 0));

		E3.set(0, 0, R3.get(0, 0));
		E3.set(0, 1, R3.get(0, 1));
		E3.set(0, 2, R3.get(0, 2));
		E3.set(1, 0, R3.get(1, 0));
		E3.set(1, 1, R3.get(1, 1));
		E3.set(1, 2, R3.get(1, 2));
		E3.set(2, 0, R3.get(2, 0));
		E3.set(2, 1, R3.get(2, 1));
		E3.set(2, 2, R3.get(2, 2));
		E3.set(0, 3, t3.get(0, 0));
		E3.set(1, 3, t3.get(1, 0));
		E3.set(2, 3, t3.get(2, 0));

		E4.set(0, 0, R4.get(0, 0));
		E4.set(0, 1, R4.get(0, 1));
		E4.set(0, 2, R4.get(0, 2));
		E4.set(1, 0, R4.get(1, 0));
		E4.set(1, 1, R4.get(1, 1));
		E4.set(1, 2, R4.get(1, 2));
		E4.set(2, 0, R4.get(2, 0));
		E4.set(2, 1, R4.get(2, 1));
		E4.set(2, 2, R4.get(2, 2));
		E4.set(0, 3, t4.get(0, 0));
		E4.set(1, 3, t4.get(1, 0));
		E4.set(2, 3, t4.get(2, 0));

		Utils.pl("E1:");
		E1.print(15, 5);
		Utils.pl("E2:");
		E2.print(15, 5);
		Utils.pl("E3:");
		E3.print(15, 5);
		Utils.pl("E4:");
		E4.print(15, 5);

		int[] scores = { 0, 0, 0, 0 };
		double[] reprojErrors = { 0, 0, 0, 0 };

		for (Correspondence2D2D c : correspondences) {

			// triangulated points
			Matrix X1 = triangulate(E1, pose, c);
			Matrix X2 = triangulate(E2, pose, c);
			Matrix X3 = triangulate(E3, pose, c);
			Matrix X4 = triangulate(E4, pose, c);

			// reprojected to second frame
			Matrix b1 = this.cameraParams.getK4x4().times(E1).times(pose).times(X1);
			Matrix b2 = this.cameraParams.getK4x4().times(E2).times(pose).times(X2);
			Matrix b3 = this.cameraParams.getK4x4().times(E3).times(pose).times(X3);
			Matrix b4 = this.cameraParams.getK4x4().times(E4).times(pose).times(X4);

			// get reprojection errors
			Matrix b1Normalized = b1.times(1 / b1.get(2, 0)).getMatrix(0, 1, 0, 0);
			Matrix b2Normalized = b2.times(1 / b2.get(2, 0)).getMatrix(0, 1, 0, 0);
			Matrix b3Normalized = b3.times(1 / b3.get(2, 0)).getMatrix(0, 1, 0, 0);
			Matrix b4Normalized = b4.times(1 / b4.get(2, 0)).getMatrix(0, 1, 0, 0);
			Matrix trueProj = new Matrix(2, 1);
			trueProj.set(0, 0, c.getX1());
			trueProj.set(1, 0, c.getY1());
			reprojErrors[0] += trueProj.minus(b1Normalized).normF();
			reprojErrors[1] += trueProj.minus(b2Normalized).normF();
			reprojErrors[2] += trueProj.minus(b3Normalized).normF();
			reprojErrors[3] += trueProj.minus(b4Normalized).normF();

			if (b1.get(2, 0) > 0) {
				Matrix a1 = pose.times(X1);
				if (a1.get(2, 0) > 0) {
					scores[0]++;
				}
			}

			if (b2.get(2, 0) > 0) {
				Matrix a2 = pose.times(X2);
				if (a2.get(2, 0) > 0) {
					scores[1]++;
				}
			}

			if (b3.get(2, 0) > 0) {
				Matrix a3 = pose.times(X3);
				if (a3.get(2, 0) > 0) {
					scores[2]++;
				}
			}

			if (b4.get(2, 0) > 0) {
				Matrix a4 = pose.times(X4);
				if (a4.get(2, 0) > 0) {
					scores[3]++;
				}
			}
		}

		// narrow down options based on chirality (should have 2 hypotheses
		// remaining)
		ArrayList<Integer> bestHypothesesInd = new ArrayList<Integer>();
		double avgScore = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;

		for (int i = 0; i < scores.length; i++) {
			int score = scores[i];
			if (score > avgScore) {
				bestHypothesesInd.add(i);
			}
		}

		// pick hypothesis based on reprojection error
		int lowestReprojInd = bestHypothesesInd.get(0);
		for (int i = 1; i < bestHypothesesInd.size(); i++) {
			if (reprojErrors[bestHypothesesInd.get(i)] < reprojErrors[lowestReprojInd]) {
				lowestReprojInd = bestHypothesesInd.get(i);
			}
		}

		// finally, set the correct decomposition
		Matrix solution = E1;
		if (lowestReprojInd == 0) {
			solution = E1;
		} else if (lowestReprojInd == 1) {
			solution = E2;
		} else if (lowestReprojInd == 2) {
			solution = E3;
		} else if (lowestReprojInd == 3) {
			solution = E4;
		}

		return solution;

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
