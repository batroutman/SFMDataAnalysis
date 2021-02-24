import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler.LegendPosition;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
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
		test4();
	}

	public static double getAngle(Correspondence2D2D c) {

		// get x and y values
		double x = c.getX1() - c.getX0();
		double y = c.getY1() - c.getY0();

		// get base angle
		double angle = Math.atan2(y, x);

		return angle;

	}

	public static void test5() {
		VirtualEnvironment mock = new VirtualEnvironment();
//		mock.generateSphericalScene(0, 1000);
		mock.getSecondaryCamera().setCz(1);
		mock.getSecondaryCamera().setCx(-0.25);
		mock.getSecondaryCamera().rotateEuler(0, Math.PI / 2, 0);

//		mock.getSecondaryCamera().setCz(-0.125);
//		mock.getSecondaryCamera().setCx(-0.25);
//		mock.getSecondaryCamera().rotateEuler(0, -0.125, 0);

		mock.generatePlanarScene(0, 1000);

		Utils.pl("numCorrespondences: " + mock.getCorrespondences().size());

		Mat image = mock.getPrimaryImage();

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
				mock.getSecondaryCamera().setCz(mock.getSecondaryCamera().getCz() + 0.01);
				image = mock.getSecondaryImage();
			} else if (c == 40) {
				// down
				mock.getSecondaryCamera().setCz(mock.getSecondaryCamera().getCz() - 0.01);
				image = mock.getSecondaryImage();
			}
		}
	}

	// create a number of mocks that gradually increase baseline and evaluate errors
	// by charting them
	public static void test4() {
		VirtualEnvironment mock = new VirtualEnvironment();
		mock.generateSphericalScene(0, 1000);

		// initial secondary camera
		double z = 0;
		double x = 0;
		double rotY = 0;
		mock.getSecondaryCamera().setCz(z);
		mock.getSecondaryCamera().setCx(x);
		mock.getSecondaryCamera().rotateEuler(0, rotY, 0);

		// ending params and motion params
		double endZ = -0.125;
		double endX = -0.25;
		double endRotY = -0.125;
		int numIterations = 100;
		double changeZ = (endZ - z) / numIterations;
		double changeX = (endX - x) / numIterations;
		double changeRotY = (endRotY - rotY) / numIterations;

		// initialize samples list
		List<Sample> samples = new ArrayList<Sample>();
		double[] indexList = new double[numIterations];
		double[] valueListFun = new double[numIterations];
		double[] valueListHom = new double[numIterations];
		double[] valueListEss = new double[numIterations];

		// deep learning training data
		INDArray input = Nd4j.zeros(numIterations, 23);
		INDArray labels = Nd4j.zeros(numIterations, 1);

		for (int i = 0; i < numIterations; i++) {

			Utils.pl("iteration: " + i);
			z += changeZ;
			x += changeX;
			rotY = changeRotY;
			mock.getSecondaryCamera().setCz(z);
			mock.getSecondaryCamera().setCx(x);
			mock.getSecondaryCamera().rotateEuler(0, rotY, 0);
			indexList[i] = (double) Math
					.round(mock.getSecondaryCamera().getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF() * 1000)
					/ 1000;
			Sample sample = new Sample();
			sample.evaluate(mock);

			valueListFun[i] = sample.transChordalEstFun;
			valueListHom[i] = sample.transChordalEstHomography;
			valueListEss[i] = sample.transChordalEstEssential;

			samples.add(sample);

			// deep learning stuff
			INDArray row = Nd4j.create(sample.correspondenceSummary.getArray());
			input.putRow(i, row);
			labels.putScalar(new int[] { i, 0 }, indexList[i] < 0.05 ? 0 : 1);
		}

		Mat image = mock.getPrimaryImage();

		// Create Chart
		final XYChart chart = new XYChartBuilder().width(600).height(400)
				.title("Error Metric Over Baseline for Planar Scene").xAxisTitle("Baseline Length")
				.yAxisTitle("Normalized Translational Chordal Distance").build();

		// Customize Chart
		chart.getStyler().setLegendPosition(LegendPosition.InsideNE);

		// Series
		chart.addSeries("Fundamental Matrix Estimate (7PA)", indexList, valueListFun);
		chart.addSeries("Homography Estimate (4PA)", indexList, valueListHom);
		chart.addSeries("Essential Matrix Estimate (5PA)", indexList, valueListEss);

		// Show it
//		new SwingWrapper(chart).displayChart();

		// train a model on this data
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1)).seed(0)
				.weightInit(WeightInit.XAVIER).list().layer(new OutputLayer.Builder(LossFunction.XENT)
						.activation(Activation.SIGMOID).nIn(23).nOut(1).build())
				.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(100));

		// train model
		DataSet data = new DataSet(input, labels);
		for (int i = 0; i < 200000; i++) {
			model.fit(data);
		}

		// create output for every training sample
		INDArray output = model.output(data.getFeatures());
		System.out.println(output);

		// let Evaluation prints stats how often the right output had the highest value
		Evaluation eval = new Evaluation();
		eval.eval(data.getLabels(), output);
		System.out.println(eval.stats());
	}

	public static void test3() {

		System.out.println("Hello TensorFlow " + TensorFlow.version());

		MultiLayerConfiguration config = new NeuralNetConfiguration.Builder().list().layer(new OutputLayer.Builder()
				.nIn(5).nOut(1).activation(Activation.SIGMOID).lossFunction(LossFunction.MSE).build()).build();
		MultiLayerNetwork model = new MultiLayerNetwork(config);

		double[] xData = new double[] { 0.0, 1.0, 2.0 };
		double[] yData = new double[] { 2.0, 1.0, 0.0 };

		// Create Chart
		XYChart chart = QuickChart.getChart("Sample Chart", "X", "Y", "y(x)", xData, yData);

		// Show it
		new SwingWrapper(chart).displayChart();

		Utils.pl("Done.");

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
