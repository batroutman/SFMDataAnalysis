import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class ComputerVision {

	// Tomono method
	public static double getD(List<Correspondence2D2D> correspondences) {

		List<Correspondence2D2D> normCorr = normalizeCorrespondences(correspondences);
		long start = System.currentTimeMillis();
		Matrix A = new Matrix(normCorr.size(), 9);
		for (int i = 0; i < normCorr.size(); i++) {
			Correspondence2D2D c = normCorr.get(i);
			double u1 = c.getX0();
			double uk = c.getX1();
			double v1 = c.getY0();
			double vk = c.getY1();

			A.set(i, 0, u1 * uk);
			A.set(i, 1, u1 * vk);
			A.set(i, 2, u1);
			A.set(i, 3, v1 * uk);
			A.set(i, 4, v1 * vk);
			A.set(i, 5, v1);
			A.set(i, 6, uk);
			A.set(i, 7, vk);
			A.set(i, 8, 1);

		}

		SingularValueDecomposition svd = A.svd();
		long end = System.currentTimeMillis();
//		Utils.pl("tomono time: " + (end - start) + "ms");
//		Utils.pl("svd S");
//		svd.getS().print(10, 5);

		double s7 = svd.getS().get(6, 6);
		double s8 = svd.getS().get(7, 7);
		double D = Math.sqrt(s7 * s8);

		return D;

	}

	public static List<Correspondence2D2D> normalizeCorrespondences(List<Correspondence2D2D> correspondences) {
		List<Correspondence2D2D> normCorr = new ArrayList<Correspondence2D2D>();
		CameraParams cameraParams = new CameraParams();

		for (int i = 0; i < correspondences.size(); i++) {
			Correspondence2D2D c = correspondences.get(i);
			Correspondence2D2D normC = new Correspondence2D2D();
			normC.setX0(c.getX0() / cameraParams.width);
			normC.setX1(c.getX1() / cameraParams.width);
			normC.setY0(c.getY0() / cameraParams.height);
			normC.setY1(c.getY1() / cameraParams.height);
			normCorr.add(normC);
		}

		return normCorr;
	}

	public static Mat estimateHomography(List<Correspondence2D2D> correspondences) {

		ArrayList<Point> matchedKeyframePoints = new ArrayList<Point>();
		ArrayList<Point> matchedPoints = new ArrayList<Point>();

//		List<Correspondence2D2D> normCorr = normalizeCorrespondences(correspondences);

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

	public static Matrix getPoseFromHomography(Mat homography, Pose primaryCamera, CameraParams cameraParams,
			List<Correspondence2D2D> correspondences) {
		return getPoseFromHomography(homography, primaryCamera, cameraParams, correspondences, new ArrayList<Matrix>());
	}

	public static Matrix getPoseFromHomography(Mat homography, Pose primaryCamera, CameraParams cameraParams,
			List<Correspondence2D2D> correspondences, List<Matrix> cheiralityPoses) {

		Mat intrinsics = cameraParams.getKMat();
		List<Mat> rotations = new ArrayList<Mat>();
		List<Mat> translations = new ArrayList<Mat>();
		List<Mat> normals = new ArrayList<Mat>();
		Calib3d.decomposeHomographyMat(homography, intrinsics, rotations, translations, normals);

		Matrix E = selectHomographySolution(primaryCamera, cameraParams, rotations, translations, correspondences,
				cheiralityPoses);
		return E;
	}

	public static Matrix selectHomographySolution(Pose primaryCamera, CameraParams cameraParams, List<Mat> rotations,
			List<Mat> translations, List<Correspondence2D2D> correspondences) {
		return selectHomographySolution(primaryCamera, cameraParams, rotations, translations, correspondences,
				new ArrayList<Matrix>());
	}

	public static Matrix selectHomographySolution(Pose primaryCamera, CameraParams cameraParams, List<Mat> rotations,
			List<Mat> translations, List<Correspondence2D2D> correspondences, List<Matrix> cheiralityPoses) {
		cheiralityPoses.clear();

//		Utils.pl("rotations: ");
//		for (int i = 0; i < rotations.size(); i++) {
//			Utils.printMatrix(rotations.get(i));
//			Utils.pl("");
//		}
//
//		Utils.pl("translations: ");
//		for (int i = 0; i < translations.size(); i++) {
//			Utils.printMatrix(translations.get(i));
//			Utils.pl("");
//		}

		if (rotations.size() == 1) {
			Matrix R = Utils.MatToMatrix(rotations.get(0));
			Matrix t = Utils.MatToMatrix(translations.get(0));
			Matrix E = Matrix.identity(4, 4);
			E.set(0, 0, R.get(0, 0));
			E.set(0, 1, R.get(0, 1));
			E.set(0, 2, R.get(0, 2));
			E.set(1, 0, R.get(1, 0));
			E.set(1, 1, R.get(1, 1));
			E.set(1, 2, R.get(1, 2));
			E.set(2, 0, R.get(2, 0));
			E.set(2, 1, R.get(2, 1));
			E.set(2, 2, R.get(2, 2));
			E.set(0, 3, t.get(0, 0));
			E.set(1, 3, t.get(1, 0));
			E.set(2, 3, t.get(2, 0));
			return E;
		}
		Matrix pose = primaryCamera.getHomogeneousMatrix();
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

		List<Matrix> poses = new ArrayList<Matrix>();
		poses.add(E1);
		poses.add(E2);
		poses.add(E3);
		poses.add(E4);

		int[] scores = { 0, 0, 0, 0 };
		double[] reprojErrors = { 0, 0, 0, 0 };

		for (Correspondence2D2D c : correspondences) {

			// triangulated points
			Matrix X1 = triangulate(E1.times(pose), pose, cameraParams, c);
			Matrix X2 = triangulate(E2.times(pose), pose, cameraParams, c);
			Matrix X3 = triangulate(E3.times(pose), pose, cameraParams, c);
			Matrix X4 = triangulate(E4.times(pose), pose, cameraParams, c);

			// reprojected to second frame
			Matrix b1 = cameraParams.getK4x4().times(E1).times(pose).times(X1);
			Matrix b2 = cameraParams.getK4x4().times(E2).times(pose).times(X2);
			Matrix b3 = cameraParams.getK4x4().times(E3).times(pose).times(X3);
			Matrix b4 = cameraParams.getK4x4().times(E4).times(pose).times(X4);

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

//		Utils.pl("scores:   " + scores[0] + ", " + scores[1] + ", " + scores[2] + ", " + scores[3]);

		// narrow down options based on cheirality (should have 2 hypotheses
		// remaining)
		ArrayList<Integer> bestHypothesesInd = new ArrayList<Integer>();
		double avgScore = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;

		for (int i = 0; i < scores.length; i++) {
			int score = scores[i];
			if (score > avgScore) {
				bestHypothesesInd.add(i);
				cheiralityPoses.add(poses.get(i));
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

	public static Matrix triangulate(Matrix secondaryPose, Matrix primaryPose, CameraParams cameraParams,
			Correspondence2D2D c) {

		Matrix Pprime = cameraParams.getK4x4().times(secondaryPose);

		Matrix P = cameraParams.getK4x4().times(primaryPose);

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

	public static Matrix triangulate(Matrix secondaryPose, Matrix primaryPose, CameraParams cameraParams,
			Correspondence2D2D c, Dbl parallax, Dbl reprojErrorSecondary, Dbl reprojErrorPrimary, Dbl wSecondary,
			Dbl wPrimary) {

//		Matrix Pprime = E.times(pose);
		Matrix Pprime = cameraParams.getK4x4().times(secondaryPose);

		Matrix P = cameraParams.getK4x4().times(primaryPose);

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

		// set parallax
		Matrix u = X.getMatrix(0, 2, 0, 0).minus(
				primaryPose.getMatrix(0, 2, 0, 2).transpose().times(primaryPose.getMatrix(0, 2, 3, 3).times(-1)));
		Matrix v = X.getMatrix(0, 2, 0, 0).minus(
				secondaryPose.getMatrix(0, 2, 0, 2).transpose().times(secondaryPose.getMatrix(0, 2, 3, 3).times(-1)));
		double cosParallax = u.transpose().times(v).get(0, 0) / (u.normF() * v.normF());

		double parallaxDegrees = Math.acos(cosParallax) * 180 / Math.PI;

		parallax.setValue(parallaxDegrees);

		// get reprojection errors
		Matrix p0 = new Matrix(4, 1);
		p0.set(0, 0, c.getX0());
		p0.set(1, 0, c.getY0());
		p0.set(2, 0, 1);
		p0.set(3, 0, 0);
		Matrix p1 = new Matrix(4, 1);
		p1.set(0, 0, c.getX1());
		p1.set(1, 0, c.getY1());
		p1.set(2, 0, 1);
		p1.set(3, 0, 0);

		Matrix proj0 = P.times(X);
		wPrimary.setValue(proj0.get(2, 0));
		proj0 = proj0.times(1 / proj0.get(2, 0));
		double reproj0 = p0.minus(proj0).normF();

		Matrix proj1 = Pprime.times(X);
		wSecondary.setValue(proj1.get(2, 0));
		proj1 = proj1.times(1 / proj1.get(2, 0));
		double reproj1 = p1.minus(proj1).normF();

		reprojErrorPrimary.setValue(reproj0);
		reprojErrorSecondary.setValue(reproj1);

		return X;
	}

	public static Matrix estimateFundamentalMatrix(List<Correspondence2D2D> correspondences) {

		CameraParams params = new CameraParams();

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
		double reprojThresh = 3;
		Mat fundamentalMatrix = Calib3d.findFundamentalMat(points0Mat, points1Mat, Calib3d.FM_RANSAC, reprojThresh, 0.9,
				5000);

		// filter out outliers (unused)
		Matrix RANSACFun = Utils.MatToMatrix(fundamentalMatrix);
		List<Point> points0Inliers = new ArrayList<Point>();
		List<Point> points1Inliers = new ArrayList<Point>();
		for (int i = 0; i < points0.size(); i++) {
			Matrix p0 = new Matrix(3, 1);
			p0.set(0, 0, points0.get(i).x);
			p0.set(1, 0, points0.get(i).y);
			p0.set(2, 0, 1);
			Matrix p1 = new Matrix(3, 1);
			p1.set(0, 0, points1.get(i).x);
			p1.set(1, 0, points1.get(i).y);
			p1.set(2, 0, 1);
			double eval = p0.transpose().times(RANSACFun).times(p1).get(0, 0);

			if (eval < reprojThresh * 1 || true) {
				points0Inliers.add(points0.get(i));
				points1Inliers.add(points1.get(i));
			}
		}

		Utils.pl("num inliers: " + points0Inliers.size());

		if (points0Inliers.size() >= 8) {
			MatOfPoint2f points0InlierMat = new MatOfPoint2f();
			MatOfPoint2f points1InlierMat = new MatOfPoint2f();
			points0InlierMat.fromList(points0Inliers);
			points1InlierMat.fromList(points1Inliers);

			fundamentalMatrix = Calib3d.findFundamentalMat(points0InlierMat, points1InlierMat, Calib3d.FM_8POINT);
			long end = System.currentTimeMillis();
//			Utils.pl("Fundamental matrix estimation time: " + (end - start) + "ms");
		}

		return Utils.MatToMatrix(fundamentalMatrix);
	}

	public static Matrix estimateEssentialMatrix(List<Correspondence2D2D> correspondences, CameraParams cameraParams) {

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
		Mat essentialMat = Calib3d.findEssentialMat(points0Mat, points1Mat, cameraParams.getKMat());
//		Mat fundamentalMatrix = Calib3d.findFundamentalMat(points0Mat, points1Mat, Calib3d.FM_RANSAC, 2, 0.99, 500);
		long end = System.currentTimeMillis();

		return Utils.MatToMatrix(essentialMat);
	}

	public static Matrix getPoseFromFundamentalMatrix(Matrix fundamentalMatrix, CameraParams cameraParams,
			List<Correspondence2D2D> correspondences) {

		// convert to essential matrix
		Matrix essentialMatrix = cameraParams.getK().transpose().times(fundamentalMatrix).times(cameraParams.getK());

		return getPoseFromEssentialMatrix(essentialMatrix, cameraParams, correspondences);

	}

	public static Matrix getPoseFromEssentialMatrix(Matrix essentialMatrix, CameraParams cameraParams,
			List<Correspondence2D2D> correspondences) {

		// convert to essential matrix
		Mat E = Utils.MatrixToMat(essentialMatrix);

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
			Matrix point3DR1t1 = triangulate(R1t1, I, cameraParams, c);
			Matrix point3DR1t2 = triangulate(R1t2, I, cameraParams, c);
			Matrix point3DR2t1 = triangulate(R2t1, I, cameraParams, c);
			Matrix point3DR2t2 = triangulate(R2t2, I, cameraParams, c);

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
						"UH OH! More than one pose passed acceptance criteria in fundamental matrix initialization! (ComputerVision::getPoseFromEssentialMatrix()) ==> numSelected: "
								+ numSelected);
			}

		}

//		Utils.pl("scores:   " + scores[0] + ", " + scores[1] + ", " + scores[2] + ", " + scores[3]);

		// find highest scoring pose
		int highestInd = 0;
		for (int i = 1; i < scores.length; i++) {
			highestInd = scores[i] > scores[highestInd] ? i : highestInd;
		}

		// convert to quaternion and pose object
		Matrix selection = possiblePoses[highestInd];

		return selection;

	}

	public static List<Matrix> triangulateCorrespondences(Matrix pose1, Matrix pose0, CameraParams cameraParams,
			List<Correspondence2D2D> correspondences) {
		List<Matrix> points = new ArrayList<Matrix>();

		for (int i = 0; i < correspondences.size(); i++) {
			Matrix point = triangulate(pose1, pose0, cameraParams, correspondences.get(i));
			points.add(point);
		}

		return points;
	}

	public static double getTotalReprojectionError(Matrix pose1, Matrix pose0, CameraParams cameraParams,
			List<Correspondence2D2D> correspondences, List<Matrix> estimatedPoints) {

		Matrix K4x4 = cameraParams.getK4x4();

		List<Matrix> truePoints0 = new ArrayList<Matrix>();
		List<Matrix> truePoints1 = new ArrayList<Matrix>();

		for (int i = 0; i < correspondences.size(); i++) {
			Matrix point0 = new Matrix(3, 1);
			point0.set(0, 0, correspondences.get(i).getX0());
			point0.set(1, 0, correspondences.get(i).getY0());
			point0.set(2, 0, 1);
			truePoints0.add(point0);

			Matrix point1 = new Matrix(3, 1);
			point1.set(0, 0, correspondences.get(i).getX1());
			point1.set(1, 0, correspondences.get(i).getY1());
			point1.set(2, 0, 1);
			truePoints1.add(point1);
		}

		double totalError0 = 0;
		double totalError1 = 0;

		for (int i = 0; i < estimatedPoints.size(); i++) {
			Matrix p = estimatedPoints.get(i);
			Matrix estimated0 = K4x4.times(pose0).times(p);
			estimated0 = estimated0.times(1 / estimated0.get(2, 0));
			Matrix estimated1 = K4x4.times(pose1).times(p);
			estimated1 = estimated1.times(1 / estimated1.get(2, 0));

			double error0 = truePoints0.get(i).minus(estimated0.getMatrix(0, 2, 0, 0)).normF();
			double error1 = truePoints1.get(i).minus(estimated1.getMatrix(0, 2, 0, 0)).normF();

			totalError0 += error0;
			totalError1 += error1;
		}

		double totalError = totalError0 + totalError1;
		return totalError;

	}

	public static double totalReconstructionError(List<Matrix> triangulated, List<Matrix> truePoints,
			DoubleWrapper median) {

		// get scale factor for triangulated points
		double trueScale = truePoints.get(0).getMatrix(0, 2, 0, 0).minus(truePoints.get(1).getMatrix(0, 2, 0, 0))
				.normF();
		double triangulatedScale = triangulated.get(0).getMatrix(0, 2, 0, 0)
				.minus(triangulated.get(1).getMatrix(0, 2, 0, 0)).normF();
//		double trueScale = truePoints.get(0).normF();
//		double triangulatedScale = triangulated.get(0).normF();
		double scaleFactor = trueScale / triangulatedScale;
//		scaleFactor = 1;

		// collect reconstruction errors
		List<Double> reconstructionErrors = new ArrayList<Double>();
		double totalError = 0;

		for (int i = 0; i < triangulated.size(); i++) {
			Matrix scaledPoint = triangulated.get(i).getMatrix(0, 2, 0, 0).times(scaleFactor);
			double error = truePoints.get(i).getMatrix(0, 2, 0, 0).minus(scaledPoint).normF();
			reconstructionErrors.add(error);
			totalError += error;
//			Utils.pl("error: " + error);
		}

		// sort reconstruction errors and get median
		Collections.sort(reconstructionErrors);
		if (reconstructionErrors.size() > 0) {
			median.value = reconstructionErrors.get(reconstructionErrors.size() / 2);
		}

		return totalError;

	}

	public static void parallaxAndGoodPoints(Matrix primaryPose, Matrix secondaryPose, CameraParams cameraParams,
			List<Matrix> triangulatedPoints, List<Correspondence2D2D> correspondences, Dbl parallax,
			Dbl numGoodPoints) {

		double parallaxRequirement = 3.0;
		double reprojectionRequirement = 20;
		int numGoodParallax = 0;
		int numGood = 0;

		for (int i = 0; i < triangulatedPoints.size(); i++) {
			Correspondence2D2D c = correspondences.get(i);
			Matrix X = triangulatedPoints.get(i);

			// get parallax
			Matrix u = X.getMatrix(0, 2, 0, 0).minus(
					primaryPose.getMatrix(0, 2, 0, 2).transpose().times(primaryPose.getMatrix(0, 2, 3, 3).times(-1)));
			Matrix v = X.getMatrix(0, 2, 0, 0).minus(secondaryPose.getMatrix(0, 2, 0, 2).transpose()
					.times(secondaryPose.getMatrix(0, 2, 3, 3).times(-1)));
			double cosParallax = u.transpose().times(v).get(0, 0) / (u.normF() * v.normF());
			double parallaxDegrees = Math.acos(cosParallax) * 180 / Math.PI;

			numGoodParallax += parallaxDegrees > parallaxRequirement ? 1 : 0;

			// get reprojection errors
			Matrix p0 = new Matrix(3, 1);
			p0.set(0, 0, c.getX0());
			p0.set(1, 0, c.getY0());
			p0.set(2, 0, 1);
			Matrix p1 = new Matrix(3, 1);
			p1.set(0, 0, c.getX1());
			p1.set(1, 0, c.getY1());
			p1.set(2, 0, 1);

			Matrix proj0 = cameraParams.getK4x4().times(primaryPose).times(X);
			double wPrimary = proj0.get(2, 0);
			proj0 = proj0.times(1 / wPrimary);
			double reproj0 = p0.minus(proj0.getMatrix(0, 2, 0, 0)).normF();

			Matrix proj1 = cameraParams.getK4x4().times(secondaryPose).times(X);
			double wSecondary = proj1.get(2, 0);
			proj1 = proj1.times(1 / wSecondary);
			double reproj1 = p1.minus(proj1.getMatrix(0, 2, 0, 0)).normF();

			// check if good point
			numGood += wPrimary > 0 && wSecondary > 0 && reproj0 < reprojectionRequirement
					&& reproj1 < reprojectionRequirement ? 1 : 0;

		}

		parallax.setValue(numGoodParallax);
		numGoodPoints.setValue(numGood);

	}

}
