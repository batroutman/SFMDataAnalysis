import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import Jama.Matrix;

public class TUMAnalyzer {

	public static CameraParams cameraParams = new CameraParams();

	public static void generateTestData(String tumFilePath, int batchSize) {

		String OUT_FILE = "results/data/TUM_samples_" + batchSize + ".dat";

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

		// for each batch,
		for (int i = 0; i < batches.size(); i++) {
			// // get orb features of first frame
			ImageData imgData0 = new ImageData(batches.get(i).get(0).getRawFrame());
			imgData0.detectAndComputeORB();
			Pose pose0 = poses.get(i * batchSize);

			// // iterate through other frames
			for (int j = 1; j < batches.get(i).size(); j++) {

				// // // get orb features of frame and match them to first frame
				ImageData imgData1 = new ImageData(batches.get(i).get(j).getRawFrame());
				imgData1.detectAndComputeORB();
				List<Correspondence2D2D> correspondences = ImageData.matchDescriptors(imgData0.getKeypoints().toList(),
						imgData0.getDescriptors(), imgData1.getKeypoints().toList(), imgData1.getDescriptors());

				// visualize matches
				Mat dest = imgData1.image.clone();
				Imgproc.cvtColor(dest, dest, Imgproc.COLOR_RGB2BGR);
				for (Correspondence2D2D c : correspondences) {
					Imgproc.line(dest, new Point(c.getX0(), c.getY0()), new Point(c.getX1(), c.getY1()),
							new Scalar(0, 255, 0), 1);
				}

				HighGui.imshow("Frame", dest);
				HighGui.waitKey(1);
				Utils.pl("index in batch: " + j);
				Utils.pl("num correspondences: " + correspondences.size());

				// // // create poses for first and current frames, calculate true difference
				Pose pose1 = poses.get(i * batchSize + j);
				Pose poseDiff = Utils.getPoseDifference(pose0, pose1);
				double baselineLength = Math.sqrt(
						Math.pow(poseDiff.getCx(), 2) + Math.pow(poseDiff.getCy(), 2) + Math.pow(poseDiff.getCz(), 2));
				Utils.pl("calculated baseline length: " + baselineLength);

				// // // get sample for correspondences
				Sample sample = new Sample();
				sample.evaluate(new Pose(), poseDiff, correspondences, cameraParams, new Matrix(3, 3), true);

				// // // create finalized data and add it to output string (with frame nums)
				FinalizedData fd = new FinalizedData();
				fd.summary = sample.correspondenceSummary;
				fd.totalReconstErrorEstFun = sample.totalReconstErrorEstFun;
				fd.totalReconstErrorEstHomography = sample.totalReconstErrorEstHomography;
				fd.totalReconstErrorEstEssential = sample.totalReconstErrorEstEssential;
				fd.transChordalEstFun = sample.transChordalEstFun;
				fd.transChordalEstHomography = sample.transChordalEstHomography;
				fd.transChordalEstEssential = sample.transChordalEstEssential;
				fd.baseline = Math.sqrt(
						Math.pow(poseDiff.getCx(), 2) + Math.pow(poseDiff.getCy(), 2) + Math.pow(poseDiff.getCz(), 2));

				Utils.pl("avg reconstruction error: " + (fd.totalReconstErrorEstFun / fd.summary.numCorrespondences));
				Utils.pl("transChordalEstFun: " + fd.transChordalEstFun);

				output += "# " + (i * batchSize) + "," + (j + i * batchSize) + "\n";
				output += fd.stringify();

				Utils.pl("calculated pose: ");
				sample.poseEstFun.print(15, 5);

				Utils.pl("poseDiff: ");
				poseDiff.getHomogeneousMatrix().print(15, 5);

				Utils.pl("poseDiff radians: ");
				Utils.pl("x: " + poseDiff.getRotX());
				Utils.pl("y: " + poseDiff.getRotY());
				Utils.pl("z: " + poseDiff.getRotZ());

				Utils.pl("poseDiff quaternion: ");
				poseDiff.getQuaternion().print(15, 10);

			}

		}

//		Utils.pl(output);
		// save output string
		try {

//			FileWriter fw = new FileWriter(OUT_FILE);
//			fw.write(output);
//			fw.close();

		} catch (Exception e) {

		}
		Utils.pl("end of function.");
	}

}
