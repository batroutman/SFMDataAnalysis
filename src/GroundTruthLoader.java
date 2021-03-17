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

		try {
			BufferedReader br = new BufferedReader(new FileReader(datasetPath + "groundtruth.txt"));
			String line = "";
//			Utils.pl("groundtruth: ");
			while (line != null) {
				line = line.trim();
				if (line.length() != 0 && line.charAt(0) != '#') {

					// construct pose
					String[] tokens = line.split(" ");
					double tx = Double.parseDouble(tokens[1]);
					double ty = Double.parseDouble(tokens[2]);
					double tz = Double.parseDouble(tokens[3]);
					double qx = Double.parseDouble(tokens[4]);
					double qy = Double.parseDouble(tokens[5]);
					double qz = Double.parseDouble(tokens[6]);
					double qw = Double.parseDouble(tokens[7]);

//					Utils.pl("tx: " + tx + ", ty: " + ty + ", tz: " + tz + ", qx: " + qx + ", qy: " + qy + ", qz: " + qz
//							+ ", qw: " + qw);

					Pose pose = new Pose();
					pose.setQw(qw);
					pose.setQx(qx);
					pose.setQy(qy);
					pose.setQz(qz);
					pose.setT(tx, ty, tz);

					poses.add(pose);

				}
				line = br.readLine();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return poses;
	}

}
