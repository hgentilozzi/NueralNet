package nnet.matrix.data;

public interface NnetBatchDataIntf {
	
	public boolean validate();
	public String getError();
	public void reset() throws Exception;
	public boolean atEof();
	public NNetBatch nextBatch() throws Exception;

}
