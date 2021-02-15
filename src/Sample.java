import java.util.ArrayList;
import java.util.List;

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

}
