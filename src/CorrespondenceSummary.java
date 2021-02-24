import java.util.ArrayList;
import java.util.List;

public class CorrespondenceSummary {

	public int numCorrespondences = 0;
	public double meanDisparity = 0;
	public double stdDevDisparity = 0;
	public double minX0 = 999999999;
	public double maxX0 = 0;
	public double minY0 = 999999999;
	public double maxY0 = 0;
	public double minX1 = 999999999;
	public double maxX1 = 0;
	public double minY1 = 999999999;
	public double maxY1 = 0;
	public double rangeX0 = 0;
	public double rangeY0 = 0;
	public double rangeX1 = 0;
	public double rangeY1 = 0;

	// rotation bins (north, northeast, east, etc.)
	public int binN = 0;
	public int binNE = 0;
	public int binE = 0;
	public int binSE = 0;
	public int binS = 0;
	public int binSW = 0;
	public int binW = 0;
	public int binNW = 0;

	public CorrespondenceSummary() {
	}

	public CorrespondenceSummary(List<Correspondence2D2D> correspondences) {
		this.evaluate(correspondences);
	}

	// return a double array of the input data (normalized bins)
	public double[] getArray() {
		double[] features = new double[23];
		features[0] = this.numCorrespondences / 300.0;
		features[1] = this.meanDisparity / 800.0;
		features[2] = this.stdDevDisparity / 800.0;
		features[3] = this.minX0 / 640.0;
		features[4] = this.maxX0 / 640.0;
		features[5] = this.minY0 / 480.0;
		features[6] = this.maxY0 / 480.0;
		features[7] = this.minX1 / 640.0;
		features[8] = this.maxX1 / 640.0;
		features[9] = this.minY1 / 480.0;
		features[10] = this.maxY1 / 480.0;
		features[11] = this.rangeX0 / 640.0;
		features[12] = this.rangeY0 / 480.0;
		features[13] = this.rangeX1 / 640.0;
		features[14] = this.rangeY1 / 480.0;

		// get magnitude of rotation bins
		double mag = Math.sqrt(Math.pow(binN, 2) + Math.pow(binNE, 2) + Math.pow(binE, 2) + Math.pow(binSE, 2)
				+ Math.pow(binS, 2) + Math.pow(binSW, 2) + Math.pow(binW, 2) + Math.pow(binNW, 2));

		features[15] = this.binN / mag;
		features[16] = this.binNE / mag;
		features[17] = this.binE / mag;
		features[18] = this.binSE / mag;
		features[19] = this.binS / mag;
		features[20] = this.binSW / mag;
		features[21] = this.binW / mag;
		features[22] = this.binNW / mag;

		return features;
	}

	public void evaluate(List<Correspondence2D2D> correspondences) {

		this.numCorrespondences = correspondences.size();

		List<Double> disparities = new ArrayList<Double>();
		double sum = 0;

		for (int i = 0; i < correspondences.size(); i++) {
			Correspondence2D2D c = correspondences.get(i);

			// handle rotation data
			this.incrementRotationBin(this.getAngle(c));

			// handle min and max
			this.minX0 = Math.min(this.minX0, c.getX0());
			this.minY0 = Math.min(this.minY0, c.getY0());
			this.minX1 = Math.min(this.minX1, c.getX1());
			this.minY1 = Math.min(this.minY1, c.getY1());
			this.maxX0 = Math.max(this.maxX0, c.getX0());
			this.maxY0 = Math.max(this.maxY0, c.getY0());
			this.maxX1 = Math.max(this.maxX1, c.getX1());
			this.maxY1 = Math.max(this.maxY1, c.getY1());

			double disparity = Math.sqrt(Math.pow(c.getX0() - c.getX1(), 2) + Math.pow(c.getY0() - c.getY1(), 2));
			sum += disparity;
			disparities.add(disparity);

		}

		// range
		this.rangeX0 = maxX0 - minX0;
		this.rangeY0 = maxY0 - minY0;
		this.rangeX1 = maxX1 - minX1;
		this.rangeY1 = maxY1 - minY1;

		// mean
		this.meanDisparity = sum / this.numCorrespondences;

		// get standard deviation
		double sumSqDev = 0;
		for (int i = 0; i < disparities.size(); i++) {
			double disparity = disparities.get(i);
			sumSqDev += Math.pow(disparity - this.meanDisparity, 2);
		}
		this.stdDevDisparity = Math.sqrt(sumSqDev / this.numCorrespondences);

	}

	public double getAngle(Correspondence2D2D c) {

		// get x and y values
		double x = c.getX1() - c.getX0();
		double y = -(c.getY1() - c.getY0());

		// get base angle
		double angle = Math.atan2(y, x);

		return angle;

	}

	public double incrementRotationBin(double angle) {

		double newAngle = angle >= 0 ? angle + 0.3926990816987 : angle - 0.3926990816987;
		double index = (int) (newAngle / 0.7853981633974);

		if (index == 0) {
			this.binE++;
		} else if (index == 1) {
			this.binNE++;
		} else if (index == 2) {
			this.binN++;
		} else if (index == 3) {
			this.binNW++;
		} else if (index >= 4) {
			this.binW++;
		} else if (index == -1) {
			this.binSE++;
		} else if (index == -2) {
			this.binS++;
		} else if (index == -3) {
			this.binSW++;
		} else if (index <= -4) {
			this.binW++;
		}

		return index;

	}

	public void printData() {
		Utils.pl("numCorrespondences: " + numCorrespondences);
		Utils.pl("meanDisparity: " + meanDisparity);
		Utils.pl("stdDevDisparity: " + stdDevDisparity);
		Utils.pl("minX0: " + minX0);
		Utils.pl("maxX0: " + maxX0);
		Utils.pl("minY0: " + minY0);
		Utils.pl("maxY0: " + maxY0);
		Utils.pl("rangeX0: " + rangeX0);
		Utils.pl("rangeY0: " + rangeY0);
		Utils.pl("minX1: " + minX1);
		Utils.pl("maxX1: " + maxX1);
		Utils.pl("minY1: " + minY1);
		Utils.pl("maxY1: " + maxY1);
		Utils.pl("rangeX1: " + rangeX1);
		Utils.pl("rangeY1: " + rangeY1);
		Utils.pl("binN: " + binN);
		Utils.pl("binNE: " + binNE);
		Utils.pl("binE: " + binE);
		Utils.pl("binSE: " + binSE);
		Utils.pl("binS: " + binS);
		Utils.pl("binSW: " + binSW);
		Utils.pl("binW: " + binW);
		Utils.pl("binNW: " + binNW);
	}

	public String stringify() {
		String output = "";

		output += this.numCorrespondences + ",";
		output += this.meanDisparity + ",";
		output += this.stdDevDisparity + ",";
		output += this.minX0 + ",";
		output += this.maxX0 + ",";
		output += this.minY0 + ",";
		output += this.maxY0 + ",";
		output += this.rangeX0 + ",";
		output += this.rangeY0 + ",";
		output += this.minX1 + ",";
		output += this.maxX1 + ",";
		output += this.minY1 + ",";
		output += this.maxY1 + ",";
		output += this.rangeX1 + ",";
		output += this.rangeY1 + ",";
		output += this.binN + ",";
		output += this.binNE + ",";
		output += this.binE + ",";
		output += this.binSE + ",";
		output += this.binS + ",";
		output += this.binSW + ",";
		output += this.binW + ",";
		output += this.binNW + "\n";

		return output;
	}

	public static CorrespondenceSummary parse(String input) {
		CorrespondenceSummary cs = new CorrespondenceSummary();
		String[] tokens = input.split(",");

		cs.numCorrespondences = Integer.parseInt(tokens[0]);
		cs.meanDisparity = Double.parseDouble(tokens[1]);
		cs.stdDevDisparity = Double.parseDouble(tokens[2]);
		cs.minX0 = Double.parseDouble(tokens[3]);
		cs.maxX0 = Double.parseDouble(tokens[4]);
		cs.minY0 = Double.parseDouble(tokens[5]);
		cs.maxY0 = Double.parseDouble(tokens[6]);
		cs.rangeX0 = Double.parseDouble(tokens[7]);
		cs.rangeY0 = Double.parseDouble(tokens[8]);
		cs.minX1 = Double.parseDouble(tokens[9]);
		cs.maxX1 = Double.parseDouble(tokens[10]);
		cs.minY1 = Double.parseDouble(tokens[11]);
		cs.maxY1 = Double.parseDouble(tokens[12]);
		cs.rangeX1 = Double.parseDouble(tokens[13]);
		cs.rangeY1 = Double.parseDouble(tokens[14]);
		cs.binN = Integer.parseInt(tokens[15]);
		cs.binNE = Integer.parseInt(tokens[16]);
		cs.binE = Integer.parseInt(tokens[17]);
		cs.binSE = Integer.parseInt(tokens[18]);
		cs.binS = Integer.parseInt(tokens[19]);
		cs.binSW = Integer.parseInt(tokens[20]);
		cs.binW = Integer.parseInt(tokens[21]);
		cs.binNW = Integer.parseInt(tokens[22]);

		return cs;
	}

}
