
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

	// robust reconstruction criteria
	public double funNumGood = 0;
	public double funNumParallax = 0;
	public double essNumGood = 0;
	public double essNumParallax = 0;
	public double homNumGood = 0;
	public double homNumParallax = 0;

	public FinalizedData() {

	}

	public String stringify() {
		String output = this.summary.stringify();

		output += this.totalReconstErrorEstFun + "," + this.totalReconstErrorEstHomography + ","
				+ this.totalReconstErrorEstEssential + "," + this.medianReconstErrorEstFun + ","
				+ this.medianReconstErrorEstHomography + "," + this.medianReconstErrorEstEssential + ","
				+ this.transChordalEstFun + "," + this.transChordalEstHomography + "," + this.transChordalEstEssential
				+ "," + this.baseline + "," + this.funNumGood + "," + this.funNumParallax + "," + this.essNumGood + ","
				+ this.essNumParallax + "," + this.homNumGood + "," + this.homNumParallax + "\n";

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

		fd.funNumGood = Double.parseDouble(errors[10]);
		fd.funNumParallax = Double.parseDouble(errors[11]);
		fd.essNumGood = Double.parseDouble(errors[12]);
		fd.essNumParallax = Double.parseDouble(errors[13]);
		fd.homNumGood = Double.parseDouble(errors[14]);
		fd.homNumParallax = Double.parseDouble(errors[15]);

		return fd;
	}

}
