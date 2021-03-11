import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

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
				ImageData imgData1 = new ImageData(batches.get(i).get(1).getRawFrame());
				imgData1.detectAndComputeORB();
				List<Correspondence2D2D> correspondences = ImageData.matchDescriptors(imgData0.getKeypoints().toList(),
						imgData0.getDescriptors(), imgData1.getKeypoints().toList(), imgData1.getDescriptors());

				// // // create poses for first and current frames, calculate true difference
				Pose pose1 = poses.get(i * batchSize + j);
				Pose poseDiff = Utils.getPoseDifference(pose0, pose1);

				// // // get sample for correspondences
				Sample sample = new Sample();
				sample.evaluate(pose0, pose1, correspondences, cameraParams, new Matrix(3, 3), true);

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

				output += "# " + i + "," + j + "\n";
				output += fd.stringify();
			}

		}

		// save output string
		try {

			FileWriter fw = new FileWriter(OUT_FILE);
			fw.write(output);
			fw.close();

		} catch (Exception e) {

		}
	}

}
