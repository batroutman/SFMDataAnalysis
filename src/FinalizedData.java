
public class FinalizedData {

	public CorrespondenceSummary summary = null;

	// total reconstruction errors
	public double totalReconstErrorEstFun = 0;
	public double totalReconstErrorEstHomography = 0;
	public double totalReconstErrorEstEssential = 0;

	// normalized translational chordal distances
	public double transChordalEstFun = 0;
	public double transChordalEstHomography = 0;
	public double transChordalEstEssential = 0;

	public FinalizedData() {

	}

	public String stringify() {
		String output = this.summary.stringify();

		output += this.totalReconstErrorEstFun + "," + this.totalReconstErrorEstHomography + ","
				+ this.totalReconstErrorEstEssential + "," + this.transChordalEstFun + ","
				+ this.transChordalEstHomography + "," + this.transChordalEstEssential + "\n";

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
		fd.transChordalEstFun = Double.parseDouble(errors[3]);
		fd.transChordalEstHomography = Double.parseDouble(errors[4]);
		fd.transChordalEstEssential = Double.parseDouble(errors[5]);

		return fd;
	}

}
