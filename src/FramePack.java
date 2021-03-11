

import org.opencv.core.Mat;

// bundles a raw frame and its processed frame (the raw frame that has undergone some degree of preprocessing for tracking)
public class FramePack {

	private long timestamp = 0;
	private Mat rawFrame = null;
	private Mat processedFrame = null;
	private byte[] rawFrameBuffer = null;
	private byte[] processedFrameBuffer = null;

	public long getTimestamp() {
		return timestamp;
	}

	public void setTimestamp(long timestamp) {
		this.timestamp = timestamp;
	}

	public Mat getRawFrame() {
		return rawFrame;
	}

	public void setRawFrame(Mat rawFrame) {
		this.rawFrame = rawFrame;
	}

	public Mat getProcessedFrame() {
		return processedFrame;
	}

	public void setProcessedFrame(Mat processedFrame) {
		this.processedFrame = processedFrame;
	}

	public byte[] getRawFrameBuffer() {
		return rawFrameBuffer;
	}

	public void setRawFrameBuffer(byte[] rawFrameBuffer) {
		this.rawFrameBuffer = rawFrameBuffer;
	}

	public byte[] getProcessedFrameBuffer() {
		return processedFrameBuffer;
	}

	public void setProcessedFrameBuffer(byte[] processedFrameBuffer) {
		this.processedFrameBuffer = processedFrameBuffer;
	}

}
