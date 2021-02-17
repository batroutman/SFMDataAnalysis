import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.tensorflow.TensorFlow;

import Jama.Matrix;

public class SFMDataAnalysis {

	public static void main(String[] args) {
		// link OpenCV binaries
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("Hello world!");
		test3();
	}

	public static double getAngle(Correspondence2D2D c) {

		// get x and y values
		double x = c.getX1() - c.getX0();
		double y = c.getY1() - c.getY0();

		// get base angle
		double angle = Math.atan2(y, x);

		return angle;

	}

	public static void test3() {

		System.out.println("Hello TensorFlow " + TensorFlow.version());

		MultiLayerNetwork model = new MultiLayerNetwork(config);

	}

	public static void test2() {
		Correspondence2D2D c = new Correspondence2D2D();
		c.setX0(0);
		c.setX1(1);
		c.setY0(0);
		c.setY1(-0.4142);

		double angle = getAngle(c);
		Utils.pl("angle: " + angle);

		double newAngle = angle >= 0 ? angle + 0.3926990816987 : angle - 0.3926990816987;
		double result = newAngle / 0.7853981633974;

		Utils.pl("result: " + result);
	}

	public static void test() {

		Double a = Double.parseDouble("3.5674567765E-3\n");
		Utils.pl("a: " + a);

		VirtualEnvironment mock = new VirtualEnvironment();
		mock.getSecondaryCamera().setCz(-1);
//		mock.getSecondaryCamera().setCx(-0.5);
//		mock.getSecondaryCamera().rotateEuler(0, -0.1, 0);
//		mock.getSecondaryCamera().rotateEuler(0, 0.1, 0);
		mock.generatePoints(0, 1000, -10, 10, -10, 10, -10, 10);

		List<Correspondence2D2D> correspondences = mock.getCorrespondences();
		Utils.pl("number of correspondences: " + correspondences.size());
		for (Correspondence2D2D corr : correspondences) {
//			Utils.pl(corr.getX0() + ", " + corr.getY0() + "  ====>    " + corr.getX1() + ", " + corr.getY1());
		}

		Mat image = mock.getPrimaryImage();
		Matrix trueFun = mock.getTrueFundamentalMatrix();

		Matrix estFun = ComputerVision.estimateFundamentalMatrix(correspondences);
		Mat homography = ComputerVision.estimateHomography(correspondences);
		Matrix truePose = ComputerVision.getPoseFromFundamentalMatrix(trueFun, mock.getCameraParams(), correspondences);
		Matrix estPose = ComputerVision.getPoseFromFundamentalMatrix(estFun, mock.getCameraParams(), correspondences);
		Matrix estPoseHomography = ComputerVision.getPoseFromHomography(homography, mock.getPrimaryCamera(),
				mock.getCameraParams(), correspondences);

		Utils.pl("Calculated fundamental matrix: ");
		trueFun.print(50, 30);

		Utils.pl("Estimated fundamental matrix: ");
		estFun.print(50, 30);

		Utils.pl("Estimated homography: ");
		Utils.MatToMatrix(homography).print(50, 30);

		Utils.pl("Pose (calculated from true fundamental matrix): ");
		truePose.print(50, 30);

		Utils.pl("Pose (calculated from estimated fundamental matrix): ");
		estPose.print(50, 30);

		Utils.pl("Pose (calculated from estimated homography): ");
		estPoseHomography.print(50, 30);

		Utils.pl("Absolute true pose: ");
		mock.getSecondaryCamera().getHomogeneousMatrix().print(50, 30);

		Matrix evalMatrix = estPose;

		List<Matrix> estimatedPoints = ComputerVision.triangulateCorrespondences(evalMatrix,
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences);

		double error = ComputerVision.getTotalReprojectionError(evalMatrix,
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences,
				estimatedPoints);

		Utils.pl("reprojectionError: " + error);
		Utils.pl("avg reprojectionError: " + (error / correspondences.size()));

		double chordal = Utils.chordalDistance(evalMatrix.getMatrix(0, 2, 0, 2),
				mock.getSecondaryCamera().getHomogeneousMatrix().getMatrix(0, 2, 0, 2));
		Utils.pl("Chordal distance: " + chordal);

		// sample test
		Sample sample = new Sample();
		long start = System.currentTimeMillis();
		sample.evaluate(mock);
		long end = System.currentTimeMillis();
		Utils.pl("time to evaluate: " + (end - start) + "ms");
		sample.printErrors();

		String output = sample.stringify();
		Utils.pl("output: ");
		Utils.p(output);

		Sample sample2 = Sample.parse(output);
		Utils.pl("");
		Utils.pl("sample2 output: ");
		Utils.pl(sample2.stringify());

		List<Correspondence2D2D> corr = new ArrayList<Correspondence2D2D>();
		corr.add(new Correspondence2D2D(0, 0, -1, -1));
		CorrespondenceSummary summary = new CorrespondenceSummary(sample.correspondences);
		Utils.pl("");
		summary.printData();

		while (true) {
			HighGui.imshow("test", image);
			char c = (char) HighGui.waitKey(1);
			if (c == 'A') {
				image = mock.getPrimaryImage();
			} else if (c == 'D') {
				image = mock.getSecondaryImage();
			} else if (c == 37) {
				// left
				mock.getSecondaryCamera().rotateEuler(0, 0.1, 0);
				image = mock.getSecondaryImage();
			} else if (c == 39) {
				// right
				mock.getSecondaryCamera().rotateEuler(0, -0.1, 0);
				image = mock.getSecondaryImage();
			} else if (c == 38) {
				// up
				mock.getSecondaryCamera().setCz(mock.getSecondaryCamera().getCz() + 0.1);
				image = mock.getSecondaryImage();
			} else if (c == 40) {
				// down
				mock.getSecondaryCamera().setCz(mock.getSecondaryCamera().getCz() - 0.1);
				image = mock.getSecondaryImage();
			}
			// down 40
			// up 38
			// right 39
			// left 37
//			Utils.pl("c == " + (int) c);
		}

	}

}
