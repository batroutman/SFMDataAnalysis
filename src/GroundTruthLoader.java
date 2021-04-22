import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class GroundTruthLoader {

	public static List<Pose> loadGroundTruth(String datasetPath) {

		// sanitize filepath
		if (datasetPath.charAt(datasetPath.length() - 1) != '/') {
			datasetPath = datasetPath + "/";
		}

		List<Pose> poses = new ArrayList<Pose>();
		List<Pose> finalPoses = new ArrayList<Pose>();

		try {
			BufferedReader br = new BufferedReader(new FileReader(datasetPath + "groundtruth.txt"));
			String line = "";
//			Utils.pl("groundtruth: ");
			while (line != null) {
				line = line.trim();
				if (line.length() != 0 && line.charAt(0) != '#') {

					// construct pose
					String[] tokens = line.split(" ");
					long time = (long) (Double.parseDouble(tokens[0]) * 1000000);
					double tx = -Double.parseDouble(tokens[1]);
					double ty = -Double.parseDouble(tokens[2]);
					double tz = -Double.parseDouble(tokens[3]);
					double qx = Double.parseDouble(tokens[4]);
					double qy = Double.parseDouble(tokens[5]);
					double qz = Double.parseDouble(tokens[6]);
					double qw = Double.parseDouble(tokens[7]);

//					Utils.pl("tx: " + tx + ", ty: " + ty + ", tz: " + tz + ", qx: " + qx + ", qy: " + qy + ", qz: " + qz
//							+ ", qw: " + qw);

					Pose pose = new Pose();
					pose.setTimestamp(time);
					pose.setQw(qw);
					pose.setQx(qx);
					pose.setQy(qy);
					pose.setQz(qz);
					pose.setT(tx, ty, tz);
//					pose.setCx(tx);
//					pose.setCy(ty);
//					pose.setCz(tz);

					poses.add(pose);

				}
				line = br.readLine();
			}

			// rebuild poses to only include poses that match with frames
			br.close();

			// get frame IDs
			br = new BufferedReader(new FileReader(datasetPath + "rgb.txt"));
			line = "";
			List<Long> frameIDs = new ArrayList<Long>();
			while (line != null) {
				line = line.trim();
				if (line.length() != 0 && line.charAt(0) != '#') {

					String[] tokens = line.split(" ");
					long time = (long) (Double.parseDouble(tokens[0]) * 1000000);
					frameIDs.add(time);

				}

				line = br.readLine();
			}

			// construct new pose list, filtering out excess poses

			int pose = 0;
			for (int i = 0; i < frameIDs.size(); i++) {

				double distance = Double.MAX_VALUE;

				boolean keepGoing = true;
				while (keepGoing) {

					// if the pose exceeds the list of poses, back up and exit the loop
					if (pose >= poses.size()) {
						pose--;
						keepGoing = false;
						continue;
					}

					// get the distance between timestamps
					double tempDist = Math.abs(poses.get(pose).getTimestamp() - frameIDs.get(i));
					if (tempDist < distance) {
						distance = tempDist;
						pose++;
					} else {
						pose--;
						keepGoing = false;
					}
				}

				finalPoses.add(poses.get(pose));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		Utils.pl("finalPoses: " + finalPoses.size());

		return finalPoses;
	}

}
