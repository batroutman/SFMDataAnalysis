import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import Jama.Matrix;

public class Sample {

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// GIVEN INFORMATION /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	public Pose primaryCamera = null;
	public Pose secondaryCamera = null;
	public List<Correspondence2D2D> correspondences = new ArrayList<Correspondence2D2D>();
	public Matrix trueFundamentalMatrix = null;

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////// CORRESPONDENCE SUMMARY //////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	public CorrespondenceSummary correspondenceSummary = new CorrespondenceSummary();

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// ITEMS TO ESTIMATE /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	public Matrix estimatedFundamentalMatrix = null;
	public Mat estimatedHomography = null;

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////// SECONDARY POSES TO ESTIMATE ///////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// pose derived from true fundamental matrix
	public Matrix poseTrueFun = null;

	// pose derived from estimated fundamental matrix
	public Matrix poseEstFun = null;

	// pose derived from homography
	public Matrix poseEstHomography = null;

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////// 3D POINT ESTIMATIONS ////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// points triangulated from secondaryCamera
	public List<Matrix> estPointsTrue = new ArrayList<Matrix>();

	// points triangulated from poseTrueFun
	public List<Matrix> estPointsTrueFun = new ArrayList<Matrix>();

	// points triangulated from poseEstFun
	public List<Matrix> estPointsEstFun = new ArrayList<Matrix>();

	// points triangulated from poseEstHomography
	public List<Matrix> estPointsEstHomography = new ArrayList<Matrix>();

	///////////////////////////////////////////////////////////////////////////////
	////////////////////////// REPROJECTION ERRORS ////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// sum of reprojection errors using primaryCamera, secondaryCamera, and
	// estPointsTrue
	public double totalReprojErrorTrue = 0;

	// sum of reprojection errors using primaryCamera, poseTrueFun, and
	// estPointsTrueFun
	public double totalReprojErrorTrueFun = 0;

	// sum of reprojection errors using primaryCamera, poseEstFun, and
	// estPointsEstFun
	public double totalReprojErrorEstFun = 0;

	// sum of reprojection errors using primaryCamera, poseEstHomography, and
	// estPointsEstHomography
	public double totalReprojErrorEstHomography = 0;

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// CHORDAL DISTANCES /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// ---- rotational chordal distances ----
	public double rotChordalTrueFun = 0;
	public double rotChordalEstFun = 0;
	public double rotChordalEstHomography = 0;

	// -- normalized translational chordal distances ----
	public double transChordalTrueFun = 0;
	public double transChordalEstFun = 0;
	public double transChordalEstHomography = 0;

	public Sample() {
	}

	public void evaluate(VirtualEnvironment mock) {

		// set cameras
		this.primaryCamera = mock.getPrimaryCamera();
		this.secondaryCamera = mock.getSecondaryCamera();

		// get correspondences and true fundamental matrix
		this.correspondences = mock.getCorrespondences();
		this.trueFundamentalMatrix = mock.getTrueFundamentalMatrix();

		// correspondence summary
		this.correspondenceSummary.evaluate(this.correspondences);

		// estimated fundamental matrix and homography
		this.estimatedFundamentalMatrix = ComputerVision.estimateFundamentalMatrix(this.correspondences);
		this.estimatedHomography = ComputerVision.estimateHomography(this.correspondences);

		// estimated poses
		this.poseTrueFun = ComputerVision.getPoseFromFundamentalMatrix(this.trueFundamentalMatrix,
				mock.getCameraParams(), this.correspondences);
		this.poseEstFun = ComputerVision.getPoseFromFundamentalMatrix(this.estimatedFundamentalMatrix,
				mock.getCameraParams(), this.correspondences);
		this.poseEstHomography = ComputerVision.getPoseFromHomography(this.estimatedHomography, mock.getPrimaryCamera(),
				mock.getCameraParams(), correspondences);

		// 3D point estimations
		this.estPointsTrue = ComputerVision.triangulateCorrespondences(this.secondaryCamera.getHomogeneousMatrix(),
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences);
		this.estPointsTrueFun = ComputerVision.triangulateCorrespondences(poseTrueFun,
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences);
		this.estPointsEstFun = ComputerVision.triangulateCorrespondences(poseEstFun,
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences);
		this.estPointsEstHomography = ComputerVision.triangulateCorrespondences(poseEstHomography,
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences);

		// reprojection errors
		this.totalReprojErrorTrue = ComputerVision.getTotalReprojectionError(
				this.secondaryCamera.getHomogeneousMatrix(), mock.getPrimaryCamera().getHomogeneousMatrix(),
				mock.getCameraParams(), correspondences, this.estPointsTrue);
		this.totalReprojErrorTrueFun = ComputerVision.getTotalReprojectionError(this.poseTrueFun,
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences,
				this.estPointsTrueFun);
		this.totalReprojErrorEstFun = ComputerVision.getTotalReprojectionError(this.poseEstFun,
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences,
				this.estPointsEstFun);
		this.totalReprojErrorEstHomography = ComputerVision.getTotalReprojectionError(this.poseEstHomography,
				mock.getPrimaryCamera().getHomogeneousMatrix(), mock.getCameraParams(), correspondences,
				this.estPointsEstHomography);

		// chordal distances
		this.rotChordalTrueFun = Utils.chordalDistance(poseTrueFun.getMatrix(0, 2, 0, 2),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 0, 2));
		this.rotChordalEstFun = Utils.chordalDistance(poseEstFun.getMatrix(0, 2, 0, 2),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 0, 2));
		this.rotChordalEstHomography = Utils.chordalDistance(poseEstHomography.getMatrix(0, 2, 0, 2),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 0, 2));

		this.transChordalTrueFun = Utils.chordalDistance(
				poseTrueFun.getMatrix(0, 2, 3, 3).times(1 / poseTrueFun.getMatrix(0, 2, 3, 3).normF()),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3)
						.times(1 / this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF()));
		this.transChordalEstFun = Utils.chordalDistance(
				poseEstFun.getMatrix(0, 2, 3, 3).times(1 / poseEstFun.getMatrix(0, 2, 3, 3).normF()),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3)
						.times(1 / this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF()));
		this.transChordalEstHomography = Utils.chordalDistance(
				poseEstHomography.getMatrix(0, 2, 3, 3).times(1 / poseEstHomography.getMatrix(0, 2, 3, 3).normF()),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3)
						.times(1 / this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF()));

	}

	public void printErrors() {
		Utils.pl("Total reprojection error (true pose): " + this.totalReprojErrorTrue);
		Utils.pl("Total reprojection error (true fundamental matrix): " + this.totalReprojErrorTrueFun);
		Utils.pl("Total reprojection error (estimated fundamental matrix): " + this.totalReprojErrorEstFun);
		Utils.pl("Total reprojection error (estimated homography): " + this.totalReprojErrorEstHomography);

		Utils.pl(
				"Average reprojection error (true pose): " + (this.totalReprojErrorTrue / this.correspondences.size()));
		Utils.pl("Average reprojection error (true fundamental matrix): "
				+ (this.totalReprojErrorTrueFun / this.correspondences.size()));
		Utils.pl("Average reprojection error (estimated fundamental matrix): "
				+ (this.totalReprojErrorEstFun / this.correspondences.size()));
		Utils.pl("Average reprojection error (estimated homography): "
				+ (this.totalReprojErrorEstHomography / this.correspondences.size()));

		Utils.pl("Rotational chordal distance (true fundamental matrix): " + this.rotChordalTrueFun);
		Utils.pl("Rotational chordal distance (estimated fundamental matrix): " + this.rotChordalEstFun);
		Utils.pl("Rotational chordal distance (estimated homography): " + this.rotChordalEstHomography);

		Utils.pl("Normalized translational chordal distance (true fundamental matrix): " + this.transChordalTrueFun);
		Utils.pl(
				"Normalized translational chordal distance (estimated fundamental matrix): " + this.transChordalEstFun);
		Utils.pl("Normalized translational chordal distance (estimated homography): " + this.transChordalEstHomography);
	}

	public String stringify() {
		String output = "";

		// cameras (qw,qx,qy,qz,Cx,Cy,Cz)
		Pose cam1 = this.primaryCamera;
		Pose cam2 = this.secondaryCamera;
		output += cam1.getQw() + "," + cam1.getQx() + "," + cam1.getQy() + "," + cam1.getQz() + "," + cam1.getCx() + ","
				+ cam1.getCy() + "," + cam1.getCz() + "\n";
		output += cam2.getQw() + "," + cam2.getQx() + "," + cam2.getQy() + "," + cam2.getQz() + "," + cam2.getCx() + ","
				+ cam2.getCy() + "," + cam2.getCz() + "\n";

		// correspondences
		// number of correspondences
		output += this.correspondences.size() + "\n";

		// correspondences (x0,y0,x1,y1|x0,y0,x1,y1|x0,y0,x1,y1|...)
		for (int i = 0; i < this.correspondences.size(); i++) {
			Correspondence2D2D c = this.correspondences.get(i);
			output += c.getX0() + "," + c.getY0() + "," + c.getX1() + "," + c.getY1() + "|";
		}
		output += "\n";

		// correspondence summary
		output += this.correspondenceSummary.stringify();

		// true fundamental matrix (row major order)
		for (int i = 0; i < this.trueFundamentalMatrix.getRowDimension(); i++) {
			for (int j = 0; j < this.trueFundamentalMatrix.getColumnDimension(); j++) {
				output += this.trueFundamentalMatrix.get(i, j) + ",";
			}
		}
		output += "\n";

		// estimated fundamental matrix (row major order)
		for (int i = 0; i < this.estimatedFundamentalMatrix.getRowDimension(); i++) {
			for (int j = 0; j < this.estimatedFundamentalMatrix.getColumnDimension(); j++) {
				output += this.estimatedFundamentalMatrix.get(i, j) + ",";
			}
		}
		output += "\n";

		// estimated homography (row major order)
		for (int i = 0; i < this.estimatedHomography.rows(); i++) {
			for (int j = 0; j < this.estimatedHomography.cols(); j++) {
				output += this.estimatedHomography.get(i, j)[0] + ",";
			}
		}
		output += "\n";

		// poseTrueFun matrix (row major order)
		for (int i = 0; i < this.poseTrueFun.getRowDimension(); i++) {
			for (int j = 0; j < this.poseTrueFun.getColumnDimension(); j++) {
				output += this.poseTrueFun.get(i, j) + ",";
			}
		}
		output += "\n";

		// poseEstFun matrix (row major order)
		for (int i = 0; i < this.poseEstFun.getRowDimension(); i++) {
			for (int j = 0; j < this.poseEstFun.getColumnDimension(); j++) {
				output += this.poseEstFun.get(i, j) + ",";
			}
		}
		output += "\n";

		// poseEstHomography matrix (row major order)
		for (int i = 0; i < this.poseEstHomography.getRowDimension(); i++) {
			for (int j = 0; j < this.poseEstHomography.getColumnDimension(); j++) {
				output += this.poseEstHomography.get(i, j) + ",";
			}
		}
		output += "\n";

		// estPointsTrue (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsTrue.size(); i++) {
			Matrix p = this.estPointsTrue.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// estPointsTrueFun (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsTrueFun.size(); i++) {
			Matrix p = this.estPointsTrueFun.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// estPointsEstFun (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsEstFun.size(); i++) {
			Matrix p = this.estPointsEstFun.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// estPointsEstHomography (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsEstHomography.size(); i++) {
			Matrix p = this.estPointsEstHomography.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// total reprojection errors
		// (totalReprojErrorTrue,totalReprojErrorTrueFun,totalReprojErrorEstFun,totalReprojErrorEstHomography)
		output += this.totalReprojErrorTrue + "," + this.totalReprojErrorTrueFun + "," + this.totalReprojErrorEstFun
				+ "," + this.totalReprojErrorEstHomography + "\n";

		// rotational chordal distances
		// (rotChordalTrueFun,rotChordalEstFun,rotChordalEstHomography)
		output += this.rotChordalTrueFun + "," + this.rotChordalEstFun + "," + this.rotChordalEstHomography + "\n";

		// translational chordal distances
		// (transChordalTrueFun,transChordalEstFun,transChordalEstHomography)
		output += this.transChordalTrueFun + "," + this.transChordalEstFun + "," + this.transChordalEstHomography
				+ "\n";

		return output;
	}

	public static Sample parse(String input) {
		Sample sample = new Sample();

		String[] lines = input.split("\n");
		int line = 0;

		// cameras
		// camera 1
		String[] cam1Line = lines[line].split(",");
		Pose camera1 = new Pose();
		camera1.setQw(Double.parseDouble(cam1Line[0]));
		camera1.setQx(Double.parseDouble(cam1Line[1]));
		camera1.setQy(Double.parseDouble(cam1Line[2]));
		camera1.setQz(Double.parseDouble(cam1Line[3]));
		camera1.setCx(Double.parseDouble(cam1Line[4]));
		camera1.setCy(Double.parseDouble(cam1Line[5]));
		camera1.setCz(Double.parseDouble(cam1Line[6]));
		sample.primaryCamera = camera1;
		line++;

		// camera 2
		String[] cam2Line = lines[line].split(",");
		Pose camera2 = new Pose();
		camera2.setQw(Double.parseDouble(cam2Line[0]));
		camera2.setQx(Double.parseDouble(cam2Line[1]));
		camera2.setQy(Double.parseDouble(cam2Line[2]));
		camera2.setQz(Double.parseDouble(cam2Line[3]));
		camera2.setCx(Double.parseDouble(cam2Line[4]));
		camera2.setCy(Double.parseDouble(cam2Line[5]));
		camera2.setCz(Double.parseDouble(cam2Line[6]));
		sample.secondaryCamera = camera2;
		line++;

		// correspondences
		int numCorrespondences = Integer.parseInt(lines[line]);
		List<Correspondence2D2D> correspondences = new ArrayList<Correspondence2D2D>(numCorrespondences);
		line++;

		String[] correspondenceLine = lines[line].split("\\|");
		for (int i = 0; i < correspondenceLine.length; i++) {
			String[] corr = correspondenceLine[i].split(",");
			Correspondence2D2D c = new Correspondence2D2D();
			c.setX0(Double.parseDouble(corr[0]));
			c.setY0(Double.parseDouble(corr[1]));
			c.setX1(Double.parseDouble(corr[2]));
			c.setY1(Double.parseDouble(corr[3]));
			correspondences.add(c);
		}
		sample.correspondences = correspondences;
		line++;

		// correspondence summary
		sample.correspondenceSummary = CorrespondenceSummary.parse(lines[line]);
		line++;

		// true fundamental matrix
		String[] trueFunMatLine = lines[line].split(",");
		Matrix trueFunMat = new Matrix(3, 3);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				trueFunMat.set(i, j, Double.parseDouble(trueFunMatLine[i * 3 + j]));
			}
		}
		sample.trueFundamentalMatrix = trueFunMat;
		line++;

		// estimated fundamental matrix
		String[] estFunMatLine = lines[line].split(",");
		Matrix estFunMat = new Matrix(3, 3);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				estFunMat.set(i, j, Double.parseDouble(estFunMatLine[i * 3 + j]));
			}
		}
		sample.estimatedFundamentalMatrix = estFunMat;
		line++;

		// estimated homography matrix
		String[] estHomLine = lines[line].split(",");
		Mat estHom = new Mat(3, 3, CvType.CV_64F);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				estHom.put(i, j, Double.parseDouble(estHomLine[i * 3 + j]));
			}
		}
		sample.estimatedHomography = estHom;
		line++;

		// poseTrueFun
		String[] poseTrueFunLine = lines[line].split(",");
		Matrix poseTrueFun = new Matrix(3, 4);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				poseTrueFun.set(i, j, Double.parseDouble(poseTrueFunLine[i * 3 + j]));
			}
		}
		sample.poseTrueFun = poseTrueFun;
		line++;

		// poseEstFun
		String[] poseEstFunLine = lines[line].split(",");
		Matrix poseEstFun = new Matrix(3, 4);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				poseEstFun.set(i, j, Double.parseDouble(poseEstFunLine[i * 3 + j]));
			}
		}
		sample.poseEstFun = poseEstFun;
		line++;

		// poseEstHomography
		String[] poseEstHomographyLine = lines[line].split(",");
		Matrix poseEstHomography = new Matrix(3, 4);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				poseEstHomography.set(i, j, Double.parseDouble(poseEstHomographyLine[i * 3 + j]));
			}
		}
		sample.poseEstHomography = poseEstHomography;
		line++;

		// estPointsTrue
		String[] estPointsTrueLine = lines[line].split("\\|");
		List<Matrix> estPointsTrue = new ArrayList<Matrix>(estPointsTrueLine.length);
		for (int i = 0; i < estPointsTrueLine.length; i++) {
			String[] coords = estPointsTrueLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsTrue.add(p);
		}
		sample.estPointsTrue = estPointsTrue;
		line++;

		// estPointsTrueFun
		String[] estPointsTrueFunLine = lines[line].split("\\|");
		List<Matrix> estPointsTrueFun = new ArrayList<Matrix>(estPointsTrueFunLine.length);
		for (int i = 0; i < estPointsTrueFunLine.length; i++) {
			String[] coords = estPointsTrueFunLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsTrueFun.add(p);
		}
		sample.estPointsTrueFun = estPointsTrueFun;
		line++;

		// estPointsEstFun
		String[] estPointsEstFunLine = lines[line].split("\\|");
		List<Matrix> estPointsEstFun = new ArrayList<Matrix>(estPointsEstFunLine.length);
		for (int i = 0; i < estPointsEstFunLine.length; i++) {
			String[] coords = estPointsEstFunLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsEstFun.add(p);
		}
		sample.estPointsEstFun = estPointsEstFun;
		line++;

		// estPointsEstHomography
		String[] estPointsEstHomographyLine = lines[line].split("\\|");
		List<Matrix> estPointsEstHomography = new ArrayList<Matrix>(estPointsEstHomographyLine.length);
		for (int i = 0; i < estPointsEstHomographyLine.length; i++) {
			String[] coords = estPointsEstHomographyLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsEstHomography.add(p);
		}
		sample.estPointsEstHomography = estPointsEstHomography;
		line++;

		// reprojection errors
		String[] reprojErrorLine = lines[line].split(",");
		sample.totalReprojErrorTrue = Double.parseDouble(reprojErrorLine[0]);
		sample.totalReprojErrorTrueFun = Double.parseDouble(reprojErrorLine[1]);
		sample.totalReprojErrorEstFun = Double.parseDouble(reprojErrorLine[2]);
		sample.totalReprojErrorEstHomography = Double.parseDouble(reprojErrorLine[3]);
		line++;

		// rotational chordal distances
		String[] rotChordalLine = lines[line].split(",");
		sample.rotChordalTrueFun = Double.parseDouble(rotChordalLine[0]);
		sample.rotChordalEstFun = Double.parseDouble(rotChordalLine[1]);
		sample.rotChordalEstHomography = Double.parseDouble(rotChordalLine[2]);
		line++;

		// translational chordal distances
		String[] transChordalLine = lines[line].split(",");
		sample.transChordalTrueFun = Double.parseDouble(transChordalLine[0]);
		sample.transChordalEstFun = Double.parseDouble(transChordalLine[1]);
		sample.transChordalEstHomography = Double.parseDouble(transChordalLine[2]);

		return sample;
	}

}
