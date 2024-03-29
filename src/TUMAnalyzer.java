import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.Styler.LegendPosition;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import Jama.Matrix;

public class TUMAnalyzer {

	public static CameraParams cameraParams = new CameraParams();

	public static void generateTestDataORB(String tumFilePath, int batchSize) {

		// initialize output data string
		String output = "";

		// load in the TUM data
		Buffer<FramePack> tumBuf = new TUMBuffer(tumFilePath, true);
		List<Pose> poses = GroundTruthLoader.loadGroundTruth(tumFilePath);

		// load TUM data into batches
		List<List<FramePack>> batches = new ArrayList<List<FramePack>>();
		boolean keepGoing = true;
		while (keepGoing) {
			List<FramePack> batch = new ArrayList<FramePack>();
			for (int i = 0; i < batchSize && keepGoing; i++) {
				FramePack fp = tumBuf.getNext();

				if (fp == null) {
					keepGoing = false;
					continue;
				}
				batch.add(fp);
			}

			if (batch.size() > 1) {
				batches.add(batch);
			}

		}

		int BATCH_START = 0;
		int BATCH_END = batches.size() - 0;

		String OUT_FILE = "results/data/orb-based/TUM_samples_stn_" + batchSize + ".dat";

		int numIterations = (batchSize - 1) * (BATCH_END - BATCH_START - 1) + (batches.get(BATCH_END - 1).size() - 1);
		double[] indexList = new double[numIterations];
		double[] valueListFun = new double[numIterations];
		double[] valueListHom = new double[numIterations];
		double[] valueListEss = new double[numIterations];
		double[] valueListBase = new double[numIterations];
		int chartIndex = 0;

		// for each batch,
		for (int i = BATCH_START; i < BATCH_END; i++) {
			// // get orb features of first frame
			ImageData imgData0 = new ImageData(batches.get(i).get(0).getProcessedFrame());
			imgData0.autoContrast();
			ORBExtractor extractor = new ORBExtractor(1.2f, ImageData.orb.getNLevels(), ImageData.orb.getPatchSize());
			extractor.setKeypoints(imgData0);
			imgData0.computeFeatures();

			ImageData imgData0 = new ImageData(batches.get(i).get(0).getProcessedFrame());
			MatOfPoint2f pInitial = imgData0.GFTT(1000);
			MatOfPoint2f pPrev = new MatOfPoint2f(pInitial);
			Mat prevFrame = imgData0.getImage();

			Pose pose0 = poses.get(i * batchSize);

			// // iterate through other frames
			for (int j = 1; j < batches.get(i).size(); j++) {

				Utils.pl("\n\nFrame #" + (i * batchSize + j) + "\n\n");

				// // // get orb features of frame and match them to first frame
				ImageData imgData1 = new ImageData(batches.get(i).get(j).getProcessedFrame());
				MatOfPoint2f pCurrent = new MatOfPoint2f();

				List<Correspondence2D2D> correspondences = imgData1.calcOpticalFlow(prevFrame, pInitial, pPrev,
						pCurrent);

				// create MatOfKeyPoints
				List<KeyPoint> listKeypointsNew = new ArrayList<KeyPoint>();
				List<KeyPoint> listKeypointsOld = new ArrayList<KeyPoint>();
				for (Correspondence2D2D c : correspondences) {
					KeyPoint kpNew = new KeyPoint();
					Point ptNew = new Point(c.getX1(), c.getY1());
					kpNew.pt = ptNew;
					listKeypointsNew.add(kpNew);

					KeyPoint kpOld = new KeyPoint();
					Point ptOld = new Point(c.getX0(), c.getY0());
					kpOld.pt = ptOld;
					listKeypointsOld.add(kpOld);
				}
				MatOfKeyPoint keypointsNew = new MatOfKeyPoint();
				MatOfKeyPoint keypointsOld = new MatOfKeyPoint();
				keypointsNew.fromList(listKeypointsNew);
				keypointsOld.fromList(listKeypointsOld);

				// visualize matches
				Mat dest = imgData1.image.clone();
				Imgproc.cvtColor(dest, dest, Imgproc.COLOR_GRAY2RGB);

				// black out image
//						Imgproc.rectangle(dest, new Point(0, 0), new Point(639, 479), new Scalar(0, 0, 0), -1);

				// draw keypoints
//						Features2d.drawKeypoints(dest, keypointsOld, dest, new Scalar(255, 0, 0));
				Features2d.drawKeypoints(dest, keypointsNew, dest, new Scalar(255, 0, 255));

				// draw correspondence lines
				for (Correspondence2D2D c : correspondences) {
					Imgproc.line(dest, new Point(c.getX0(), c.getY0()), new Point(c.getX1(), c.getY1()),
							new Scalar(0, 255, 0), 1);
				}

				HighGui.imshow("Frame", dest);
				HighGui.waitKey(1);
				Utils.pl("index in batch: " + j);
				Utils.pl("num correspondences: " + correspondences.size());

				// update keypoints
				prevFrame = imgData1.getImage();
				pPrev = pCurrent;

				// // // create poses for first and current frames, calculate true difference
				Pose pose1 = poses.get(i * batchSize + j);
				Utils.pl("absolute pose1: ");
				pose1.getHomogeneousMatrix().print(10, 5);
				Pose poseDiff = Utils.getPoseDifference(pose0, pose1);
				Utils.pl("poseDiff:");
				Utils.pl("rotXDeg: " + poseDiff.getRotXDeg());
				Utils.pl("rotYDeg: " + poseDiff.getRotYDeg());
				Utils.pl("rotZDeg: " + poseDiff.getRotZDeg());
				poseDiff.getHomogeneousMatrix().print(10, 5);
				double baselineLength = Math.sqrt(
						Math.pow(poseDiff.getCx(), 2) + Math.pow(poseDiff.getCy(), 2) + Math.pow(poseDiff.getCz(), 2));
				Utils.pl("calculated baseline length: " + baselineLength);

				// // // get sample for correspondences
				Sample sample = new Sample();
				sample.evaluate(new Pose(), poseDiff, correspondences, cameraParams, new Matrix(3, 3), true);
//						sample.bundleAdjust();

				// // // create finalized data and add it to output string (with frame nums)
				FinalizedData fd = new FinalizedData();
				fd.summary = sample.correspondenceSummary;
				fd.totalReconstErrorEstFun = sample.totalReconstErrorEstFun;
				fd.totalReconstErrorEstHomography = sample.totalReconstErrorEstHomography;
				fd.totalReconstErrorEstEssential = sample.totalReconstErrorEstEssential;
				fd.medianReconstErrorEstFun = sample.medianReconstErrorEstFun;
				fd.medianReconstErrorEstHomography = sample.medianReconstErrorEstHomography;
				fd.medianReconstErrorEstEssential = sample.medianReconstErrorEstEssential;
				fd.transChordalEstFun = sample.transChordalEstFun;
				fd.transChordalEstHomography = sample.transChordalEstHomography;
				fd.transChordalEstEssential = sample.transChordalEstEssential;
				fd.baseline = Math.sqrt(
						Math.pow(poseDiff.getCx(), 2) + Math.pow(poseDiff.getCy(), 2) + Math.pow(poseDiff.getCz(), 2));

				fd.funNumGood = sample.funNumGood;
				fd.funNumParallax = sample.funNumParallax;
				fd.essNumGood = sample.essNumGood;
				fd.essNumParallax = sample.essNumParallax;
				fd.homNumGood = sample.homNumGood;
				fd.homNumParallax = sample.homNumParallax;

				Utils.pl("fun mat pose estimate: ");
				sample.poseEstFun.print(10, 5);

				Utils.pl("homography pose estimate: ");
				sample.poseEstHomography.print(10, 5);

				Utils.pl("avg reconstruction error (fundamental): "
						+ (fd.totalReconstErrorEstFun / fd.summary.numCorrespondences));
				Utils.pl("median reconstruction error (fundamental): " + fd.medianReconstErrorEstFun);
				Utils.pl("transChordalEstFun: " + fd.transChordalEstFun);
				Utils.pl("");
				Utils.pl("avg reconstruction error (essential): "
						+ (fd.totalReconstErrorEstEssential / fd.summary.numCorrespondences));
				Utils.pl("median reconstruction error (essential): " + fd.medianReconstErrorEstEssential);
				Utils.pl("transChordalEstEssential: " + fd.transChordalEstEssential);
				Utils.pl("");
				Utils.pl("avg reconstruction error (homography): "
						+ (fd.totalReconstErrorEstHomography / fd.summary.numCorrespondences));
				Utils.pl("median reconstruction error (homography): " + fd.medianReconstErrorEstHomography);
				Utils.pl("transChordalEstHomography: " + fd.transChordalEstHomography);
				Utils.pl("");

				output += "# " + (i * batchSize) + "," + (j + i * batchSize) + "\n";
				output += fd.stringify();

//						int chartIndex = i * batchSize + j - 1;
				indexList[chartIndex] = chartIndex + 1;

				// average reconstruction errors
//						valueListFun[chartIndex] = sample.totalReconstErrorEstFun / sample.truePoints.size();
//						valueListHom[chartIndex] = sample.totalReconstErrorEstHomography / sample.truePoints.size();
//						valueListEss[chartIndex] = sample.totalReconstErrorEstEssential / sample.truePoints.size();

				// median reconstruction errors
//						valueListFun[chartIndex] = sample.medianReconstErrorEstFun;
//						valueListHom[chartIndex] = sample.medianReconstErrorEstHomography;
//						valueListEss[chartIndex] = sample.medianReconstErrorEstEssential;

				// translational chordal distance
//						valueListFun[chartIndex] = sample.transChordalEstFun;
//						valueListHom[chartIndex] = sample.transChordalEstHomography;
//						valueListEss[chartIndex] = sample.transChordalEstEssential;

				// parallax
//						valueListFun[chartIndex] = sample.funNumParallax;
//						valueListHom[chartIndex] = sample.homNumParallax;
//						valueListEss[chartIndex] = sample.essNumParallax;

				// num good points
//						valueListFun[chartIndex] = sample.funNumGood;
//						valueListHom[chartIndex] = sample.homNumGood;
//						valueListEss[chartIndex] = sample.essNumGood;

				// proportion good
				valueListFun[chartIndex] = sample.funNumGood / correspondences.size();
				valueListHom[chartIndex] = sample.homNumGood / correspondences.size();
				valueListEss[chartIndex] = sample.essNumGood / correspondences.size();

				valueListBase[chartIndex] = baselineLength;

				chartIndex++;

			}

		}

		boolean plot = true;
		if (plot) {

			int START = 0;
			int END = numIterations;
//					int START = (batchSize - 1) * 1;
//					int END = START + (batchSize - 1);

			// slice data
			double[] indexListSliced = Arrays.copyOfRange(indexList, START, END);
			double[] valueListFunSliced = Arrays.copyOfRange(valueListFun, START, END);
			double[] valueListHomSliced = Arrays.copyOfRange(valueListHom, START, END);
			double[] valueListEssSliced = Arrays.copyOfRange(valueListEss, START, END);
			double[] valueListBaseSliced = Arrays.copyOfRange(valueListBase, START, END);

			// Create Chart
			// Rescaled Median Reconstruction Error
			// Normalized Translational Chordal Distance
			final XYChart chart = new XYChartBuilder().width(640).height(480).theme(Styler.ChartTheme.Matlab)
					.title("Reconstruction Error on a Typical Batch of Training Data").xAxisTitle("Baseline Length")
					.yAxisTitle("Median Scaled Reconstruction Error").build();

			// Customize Chart
			chart.getStyler().setLegendPosition(LegendPosition.InsideNE);

			// Series
			chart.addSeries("Fundamental Matrix Estimate (8PA)", indexListSliced, valueListFunSliced);
			chart.addSeries("Homography Estimate (4PA)", indexListSliced, valueListHomSliced);
			chart.addSeries("Essential Matrix Estimate (5PA)", indexListSliced, valueListEssSliced);
//					chart.addSeries("Baseline Length", indexListSliced, valueListBaseSliced);
//							chart.addSeries("Tomono score", indexList, valueListTomono);
//					chart.getStyler().setYAxisMax(10.0);

			// Show it
			new SwingWrapper(chart).displayChart();
		}

//				Utils.pl(output);
		// save output string
		try {

			FileWriter fw = new FileWriter(OUT_FILE);
			fw.write(output);
			fw.close();

		} catch (Exception e) {

		}
		Utils.pl("end of function.");

	}

	public static void generateTestData(String tumFilePath, int batchSize) {

		// initialize output data string
		String output = "";

		// load in the TUM data
		Buffer<FramePack> tumBuf = new TUMBuffer(tumFilePath, true);
		List<Pose> poses = GroundTruthLoader.loadGroundTruth(tumFilePath);

		// load TUM data into batches
		List<List<FramePack>> batches = new ArrayList<List<FramePack>>();
		boolean keepGoing = true;
		while (keepGoing) {
			List<FramePack> batch = new ArrayList<FramePack>();
			for (int i = 0; i < batchSize && keepGoing; i++) {
				FramePack fp = tumBuf.getNext();

				if (fp == null) {
					keepGoing = false;
					continue;
				}
				batch.add(fp);
			}

			if (batch.size() > 1) {
				batches.add(batch);
			}

		}

		int BATCH_START = 0;
		int BATCH_END = batches.size() - 0;

		String OUT_FILE = "results/data/TUM_samples_stn_3_" + batchSize + ".dat";

		int numIterations = (batchSize - 1) * (BATCH_END - BATCH_START - 1) + (batches.get(BATCH_END - 1).size() - 1);
		double[] indexList = new double[numIterations];
		double[] valueListFun = new double[numIterations];
		double[] valueListHom = new double[numIterations];
		double[] valueListEss = new double[numIterations];
		double[] valueListBase = new double[numIterations];
		int chartIndex = 0;

		// for each batch,
		for (int i = BATCH_START; i < BATCH_END; i++) {
			// // get orb features of first frame
			ImageData imgData0 = new ImageData(batches.get(i).get(0).getProcessedFrame());
			MatOfPoint2f pInitial = imgData0.GFTT(1000);
			MatOfPoint2f pPrev = new MatOfPoint2f(pInitial);
			Mat prevFrame = imgData0.getImage();

			Pose pose0 = poses.get(i * batchSize);

			// // iterate through other frames
			for (int j = 1; j < batches.get(i).size(); j++) {

				Utils.pl("\n\nFrame #" + (i * batchSize + j) + "\n\n");

				// // // get orb features of frame and match them to first frame
				ImageData imgData1 = new ImageData(batches.get(i).get(j).getProcessedFrame());
				MatOfPoint2f pCurrent = new MatOfPoint2f();

				List<Correspondence2D2D> correspondences = imgData1.calcOpticalFlow(prevFrame, pInitial, pPrev,
						pCurrent);

				// create MatOfKeyPoints
				List<KeyPoint> listKeypointsNew = new ArrayList<KeyPoint>();
				List<KeyPoint> listKeypointsOld = new ArrayList<KeyPoint>();
				for (Correspondence2D2D c : correspondences) {
					KeyPoint kpNew = new KeyPoint();
					Point ptNew = new Point(c.getX1(), c.getY1());
					kpNew.pt = ptNew;
					listKeypointsNew.add(kpNew);

					KeyPoint kpOld = new KeyPoint();
					Point ptOld = new Point(c.getX0(), c.getY0());
					kpOld.pt = ptOld;
					listKeypointsOld.add(kpOld);
				}
				MatOfKeyPoint keypointsNew = new MatOfKeyPoint();
				MatOfKeyPoint keypointsOld = new MatOfKeyPoint();
				keypointsNew.fromList(listKeypointsNew);
				keypointsOld.fromList(listKeypointsOld);

				// visualize matches
				Mat dest = imgData1.image.clone();
				Imgproc.cvtColor(dest, dest, Imgproc.COLOR_GRAY2RGB);

				// black out image
//				Imgproc.rectangle(dest, new Point(0, 0), new Point(639, 479), new Scalar(0, 0, 0), -1);

				// draw keypoints
//				Features2d.drawKeypoints(dest, keypointsOld, dest, new Scalar(255, 0, 0));
				Features2d.drawKeypoints(dest, keypointsNew, dest, new Scalar(255, 0, 255));

				// draw correspondence lines
				for (Correspondence2D2D c : correspondences) {
					Imgproc.line(dest, new Point(c.getX0(), c.getY0()), new Point(c.getX1(), c.getY1()),
							new Scalar(0, 255, 0), 1);
				}

				HighGui.imshow("Frame", dest);
				HighGui.waitKey(1);
				Utils.pl("index in batch: " + j);
				Utils.pl("num correspondences: " + correspondences.size());

				// update keypoints
				prevFrame = imgData1.getImage();
				pPrev = pCurrent;

				// // // create poses for first and current frames, calculate true difference
				Pose pose1 = poses.get(i * batchSize + j);
				Utils.pl("absolute pose1: ");
				pose1.getHomogeneousMatrix().print(10, 5);
				Pose poseDiff = Utils.getPoseDifference(pose0, pose1);
				Utils.pl("poseDiff:");
				Utils.pl("rotXDeg: " + poseDiff.getRotXDeg());
				Utils.pl("rotYDeg: " + poseDiff.getRotYDeg());
				Utils.pl("rotZDeg: " + poseDiff.getRotZDeg());
				poseDiff.getHomogeneousMatrix().print(10, 5);
				double baselineLength = Math.sqrt(
						Math.pow(poseDiff.getCx(), 2) + Math.pow(poseDiff.getCy(), 2) + Math.pow(poseDiff.getCz(), 2));
				Utils.pl("calculated baseline length: " + baselineLength);

				// // // get sample for correspondences
				Sample sample = new Sample();
				sample.evaluate(new Pose(), poseDiff, correspondences, cameraParams, new Matrix(3, 3), true);
//				sample.bundleAdjust();

				// // // create finalized data and add it to output string (with frame nums)
				FinalizedData fd = new FinalizedData();
				fd.summary = sample.correspondenceSummary;
				fd.totalReconstErrorEstFun = sample.totalReconstErrorEstFun;
				fd.totalReconstErrorEstHomography = sample.totalReconstErrorEstHomography;
				fd.totalReconstErrorEstEssential = sample.totalReconstErrorEstEssential;
				fd.medianReconstErrorEstFun = sample.medianReconstErrorEstFun;
				fd.medianReconstErrorEstHomography = sample.medianReconstErrorEstHomography;
				fd.medianReconstErrorEstEssential = sample.medianReconstErrorEstEssential;
				fd.transChordalEstFun = sample.transChordalEstFun;
				fd.transChordalEstHomography = sample.transChordalEstHomography;
				fd.transChordalEstEssential = sample.transChordalEstEssential;
				fd.baseline = Math.sqrt(
						Math.pow(poseDiff.getCx(), 2) + Math.pow(poseDiff.getCy(), 2) + Math.pow(poseDiff.getCz(), 2));

				fd.funNumGood = sample.funNumGood;
				fd.funNumParallax = sample.funNumParallax;
				fd.essNumGood = sample.essNumGood;
				fd.essNumParallax = sample.essNumParallax;
				fd.homNumGood = sample.homNumGood;
				fd.homNumParallax = sample.homNumParallax;

				Utils.pl("fun mat pose estimate: ");
				sample.poseEstFun.print(10, 5);

				Utils.pl("homography pose estimate: ");
				sample.poseEstHomography.print(10, 5);

				Utils.pl("avg reconstruction error (fundamental): "
						+ (fd.totalReconstErrorEstFun / fd.summary.numCorrespondences));
				Utils.pl("median reconstruction error (fundamental): " + fd.medianReconstErrorEstFun);
				Utils.pl("transChordalEstFun: " + fd.transChordalEstFun);
				Utils.pl("");
				Utils.pl("avg reconstruction error (essential): "
						+ (fd.totalReconstErrorEstEssential / fd.summary.numCorrespondences));
				Utils.pl("median reconstruction error (essential): " + fd.medianReconstErrorEstEssential);
				Utils.pl("transChordalEstEssential: " + fd.transChordalEstEssential);
				Utils.pl("");
				Utils.pl("avg reconstruction error (homography): "
						+ (fd.totalReconstErrorEstHomography / fd.summary.numCorrespondences));
				Utils.pl("median reconstruction error (homography): " + fd.medianReconstErrorEstHomography);
				Utils.pl("transChordalEstHomography: " + fd.transChordalEstHomography);
				Utils.pl("");

				output += "# " + (i * batchSize) + "," + (j + i * batchSize) + "\n";
				output += fd.stringify();

//				int chartIndex = i * batchSize + j - 1;
				indexList[chartIndex] = chartIndex + 1;

				// average reconstruction errors
//				valueListFun[chartIndex] = sample.totalReconstErrorEstFun / sample.truePoints.size();
//				valueListHom[chartIndex] = sample.totalReconstErrorEstHomography / sample.truePoints.size();
//				valueListEss[chartIndex] = sample.totalReconstErrorEstEssential / sample.truePoints.size();

				// median reconstruction errors
//				valueListFun[chartIndex] = sample.medianReconstErrorEstFun;
//				valueListHom[chartIndex] = sample.medianReconstErrorEstHomography;
//				valueListEss[chartIndex] = sample.medianReconstErrorEstEssential;

				// translational chordal distance
//				valueListFun[chartIndex] = sample.transChordalEstFun;
//				valueListHom[chartIndex] = sample.transChordalEstHomography;
//				valueListEss[chartIndex] = sample.transChordalEstEssential;

				// parallax
//				valueListFun[chartIndex] = sample.funNumParallax;
//				valueListHom[chartIndex] = sample.homNumParallax;
//				valueListEss[chartIndex] = sample.essNumParallax;

				// num good points
//				valueListFun[chartIndex] = sample.funNumGood;
//				valueListHom[chartIndex] = sample.homNumGood;
//				valueListEss[chartIndex] = sample.essNumGood;

				// proportion good
				valueListFun[chartIndex] = sample.funNumGood / correspondences.size();
				valueListHom[chartIndex] = sample.homNumGood / correspondences.size();
				valueListEss[chartIndex] = sample.essNumGood / correspondences.size();

				valueListBase[chartIndex] = baselineLength;

				chartIndex++;

			}

		}

		boolean plot = true;
		if (plot) {

			int START = 0;
			int END = numIterations;
//			int START = (batchSize - 1) * 1;
//			int END = START + (batchSize - 1);

			// slice data
			double[] indexListSliced = Arrays.copyOfRange(indexList, START, END);
			double[] valueListFunSliced = Arrays.copyOfRange(valueListFun, START, END);
			double[] valueListHomSliced = Arrays.copyOfRange(valueListHom, START, END);
			double[] valueListEssSliced = Arrays.copyOfRange(valueListEss, START, END);
			double[] valueListBaseSliced = Arrays.copyOfRange(valueListBase, START, END);

			// Create Chart
			// Rescaled Median Reconstruction Error
			// Normalized Translational Chordal Distance
			final XYChart chart = new XYChartBuilder().width(640).height(480).theme(Styler.ChartTheme.Matlab)
					.title("Reconstruction Error on a Typical Batch of Training Data").xAxisTitle("Baseline Length")
					.yAxisTitle("Median Scaled Reconstruction Error").build();

			// Customize Chart
			chart.getStyler().setLegendPosition(LegendPosition.InsideNE);

			// Series
			chart.addSeries("Fundamental Matrix Estimate (8PA)", indexListSliced, valueListFunSliced);
			chart.addSeries("Homography Estimate (4PA)", indexListSliced, valueListHomSliced);
			chart.addSeries("Essential Matrix Estimate (5PA)", indexListSliced, valueListEssSliced);
//			chart.addSeries("Baseline Length", indexListSliced, valueListBaseSliced);
//					chart.addSeries("Tomono score", indexList, valueListTomono);
//			chart.getStyler().setYAxisMax(10.0);

			// Show it
			new SwingWrapper(chart).displayChart();
		}

//		Utils.pl(output);
		// save output string
		try {

			FileWriter fw = new FileWriter(OUT_FILE);
			fw.write(output);
			fw.close();

		} catch (Exception e) {

		}
		Utils.pl("end of function.");
	}

}
