
import java.util.List;

import org.ddogleg.optimization.lm.ConfigLevenbergMarquardt;

import Jama.Matrix;
import boofcv.abst.geo.bundle.BundleAdjustment;
import boofcv.abst.geo.bundle.ScaleSceneStructure;
import boofcv.abst.geo.bundle.SceneObservations;
import boofcv.abst.geo.bundle.SceneStructureMetric;
import boofcv.alg.geo.bundle.cameras.BundlePinhole;
import boofcv.factory.geo.ConfigBundleAdjustment;
import boofcv.factory.geo.FactoryMultiView;
import georegression.geometry.ConvertRotation3D_F64;
import georegression.struct.point.Vector3D_F64;
import georegression.struct.se.Se3_F64;
import georegression.struct.so.Quaternion_F64;

public class BundleAdjustor {

	// point3Ds is a list of the triangulated 3D points, correspondences is a list
	// of the correspondences found between camera0 and camera1 and must be in the
	// same order as point3Ds
	public static SceneStructureMetric bundleAdjustPair(Pose camera0, Pose camera1, List<Matrix> point3Ds,
			List<Correspondence2D2D> correspondences, int maxIterations) {

		// boofCV
		SceneStructureMetric scene = new SceneStructureMetric(false);
		scene.initialize(2, 2, point3Ds.size());
		SceneObservations observations = new SceneObservations();
		observations.initialize(2);

		// load camera poses into scene
		BundlePinhole camera = new BundlePinhole();
		CameraParams CameraIntrinsics = new CameraParams();
		camera.fx = CameraIntrinsics.fx;
		camera.fy = CameraIntrinsics.fy;
		camera.cx = CameraIntrinsics.cx;
		camera.cy = CameraIntrinsics.cy;
		camera.skew = CameraIntrinsics.s;
		for (int i = 0; i < 2; i++) {
			Pose pose = i == 0 ? camera0 : camera1;
			Se3_F64 worldToCameraGL = new Se3_F64();
			ConvertRotation3D_F64.quaternionToMatrix(pose.getQw(), pose.getQx(), pose.getQy(), pose.getQz(),
					worldToCameraGL.R);
			worldToCameraGL.T.x = pose.getTx();
			worldToCameraGL.T.y = pose.getTy();
			worldToCameraGL.T.z = pose.getTz();
			scene.setCamera(i, true, camera);
			scene.setView(i, i == 0 ? true : false, worldToCameraGL);
			scene.connectViewToCamera(i, i);
		}

		// load projected observations into observations variable
		for (int pointID = 0; pointID < correspondences.size(); pointID++) {
			Correspondence2D2D c = correspondences.get(pointID);
			float pixelX = (float) c.getX0();
			float pixelY = (float) c.getY0();
			observations.getView(0).add(pointID, pixelX, pixelY);
			pixelX = (float) c.getX1();
			pixelY = (float) c.getY1();
			observations.getView(1).add(pointID, pixelX, pixelY);
		}

		// load 3D points into scene
		for (int i = 0; i < point3Ds.size(); i++) {
			float x = (float) point3Ds.get(i).get(0, 0);
			float y = (float) point3Ds.get(i).get(1, 0);
			float z = (float) point3Ds.get(i).get(2, 0);

			scene.setPoint(i, x, y, z);
		}

		ConfigLevenbergMarquardt configLM = new ConfigLevenbergMarquardt();
		configLM.dampeningInitial = 1e-3;
		configLM.hessianScaling = true;

		ConfigBundleAdjustment configSBA = new ConfigBundleAdjustment();
		configSBA.configOptimizer = configLM;
		BundleAdjustment<SceneStructureMetric> bundleAdjustment = FactoryMultiView.bundleSparseMetric(configSBA);

		// debug
		bundleAdjustment.setVerbose(System.out, 0);

		// Specifies convergence criteria
		bundleAdjustment.configure(1e-12, 1e-12, maxIterations);

		// Scaling each variable type so that it takes on a similar numerical
		// value. This aids in optimization
		// Not important for this problem but is for others
		ScaleSceneStructure bundleScale = new ScaleSceneStructure();
		bundleScale.applyScale(scene, observations);
		bundleAdjustment.setParameters(scene, observations);

		// Runs the solver. This will take a few minutes. 7 iterations takes
		// about 3 minutes on my computer
		long startTime = System.currentTimeMillis();
		double errorBefore = bundleAdjustment.getFitScore();
		Utils.pl("error before: " + errorBefore);
		if (!bundleAdjustment.optimize(scene)) {
			throw new RuntimeException("Bundle adjustment failed?!?");
//			Utils.pl("\n\n\n\n***************************  ERROR  ****************************");
//			Utils.pl("NOTE: Bundle Adjustment failed!");
//			Utils.pl("fit score: " + bundleAdjustment.getFitScore());
//			bundleScale.undoScale(scene, observations);
//
//			Utils.pl("****************************************************************\n\n\n\n");

//			return scene;
		}

		// Print out how much it improved the model
		System.out.println();
		System.out.printf("Error reduced by %.1f%%\n", (100.0 * (errorBefore / bundleAdjustment.getFitScore() - 1.0)));
		System.out.println((System.currentTimeMillis() - startTime) / 1000.0);

		// Return parameters to their original scaling. Can probably skip this
		// step.
		bundleScale.undoScale(scene, observations);

		// load points from scene back into input
		for (int i = 0; i < scene.getPoints().size(); i++) {
			point3Ds.get(i).set(0, 0, scene.getPoints().get(i).getX());
			point3Ds.get(i).set(1, 0, scene.getPoints().get(i).getY());
			point3Ds.get(i).set(2, 0, scene.getPoints().get(i).getZ());
		}

		// load pose from scene back into input
		Se3_F64 worldToView = scene.getViews().get(1).worldToView;
		Quaternion_F64 q = ConvertRotation3D_F64.matrixToQuaternion(worldToView.getR(), null);
		q.normalize();
		Vector3D_F64 t = worldToView.getTranslation();
		camera1.setQw(q.w);
		camera1.setQx(q.x);
		camera1.setQy(q.y);
		camera1.setQz(q.z);
		camera1.setT(t.x, t.y, t.z);

		return scene;

	}

}
