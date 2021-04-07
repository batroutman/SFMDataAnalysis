
public class FinalizedData {

	public CorrespondenceSummary summary = null;

	// total reconstruction errors
	public double totalReconstErrorEstFun = 0;
	public double totalReconstErrorEstHomography = 0;
	public double totalReconstErrorEstEssential = 0;

	// total reconstruction errors
	public double medianReconstErrorEstFun = 0;
	public double medianReconstErrorEstHomography = 0;
	public double medianReconstErrorEstEssential = 0;

	// normalized translational chordal distances
	public double transChordalEstFun = 0;
	public double transChordalEstHomography = 0;
	public double transChordalEstEssential = 0;

	// other ground truth data
	public double baseline = 0;

	public FinalizedData() {

	}

	public String stringify() {
		String output = this.summary.stringify();

		output += this.totalReconstErrorEstFun + "," + this.totalReconstErrorEstHomography + ","
				+ this.totalReconstErrorEstEssential + "," + this.medianReconstErrorEstFun + ","
				+ this.medianReconstErrorEstHomography + "," + this.medianReconstErrorEstEssential + ","
				+ this.transChordalEstFun + "," + this.transChordalEstHomography + "," + this.transChordalEstEssential
				+ "," + this.baseline + "\n";

		return output;
	}

	public static FinalizedData parse(String input) {
		String[] lines = input.split("\n");
		FinalizedData fd = new FinalizedData();
		fd.summary = CorrespondenceSummary.parse(lines[0]);

		String[] errors = lines[1].split(",");
		fd.totalReconstErrorEstFun = Double.parseDouble(errors[0]);
		fd.totalReconstErrorEstHomography = Double.parseDouble(errors[1]);
		fd.totalReconstErrorEstEssential = Double.parseDouble(errors[2]);
		fd.medianReconstErrorEstFun = Double.parseDouble(errors[3]);
		fd.medianReconstErrorEstHomography = Double.parseDouble(errors[4]);
		fd.medianReconstErrorEstEssential = Double.parseDouble(errors[5]);
		fd.transChordalEstFun = Double.parseDouble(errors[6]);
		fd.transChordalEstHomography = Double.parseDouble(errors[7]);
		fd.transChordalEstEssential = Double.parseDouble(errors[8]);
		fd.baseline = Double.parseDouble(errors[9]);

		return fd;
	}

}
