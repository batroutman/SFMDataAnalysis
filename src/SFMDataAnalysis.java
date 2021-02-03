import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;

public class SFMDataAnalysis {

	public static void main(String[] args) {
		// link OpenCV binaries
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("Hello world!");
		test();
	}

	public static void test() {

		VirtualEnvironment mock = new VirtualEnvironment();
		mock.getSecondaryCamera().setCx(1);
		mock.generatePoints(0, 1000, -10, 10, -10, 10, -10, 10);

		List<Correspondence2D2D> correspondences = mock.getCorrespondences();
		Utils.pl("number of correspondences: " + correspondences.size());
		for (Correspondence2D2D corr : correspondences) {
			Utils.pl(corr.getX0() + ", " + corr.getY0() + "  ====>    " + corr.getX1() + ", " + corr.getY1());
		}

		Mat image = mock.getPrimaryImage();

		while (true) {
			HighGui.imshow("test", image);
			char c = (char) HighGui.waitKey(1);
			if (c == 'A') {
				image = mock.getPrimaryImage();
			} else if (c == 'D') {
				image = mock.getSecondaryImage();
			}
		}

	}

}
