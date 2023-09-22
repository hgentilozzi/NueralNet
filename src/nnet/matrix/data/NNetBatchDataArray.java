package nnet.matrix.data;

import java.util.Arrays;

import nnet.matrix.*;

/**
 * A batch based on a set of arrays
 */
public class NNetBatchDataArray implements NnetBatchDataIntf {
	private double[][] inputData;
	private double[][] expectedResults;
	private int currIdx;
	private String error;


	public NNetBatchDataArray(double[][] inputData, double[][] expectedResukts) {
		this.inputData = inputData;
		this.expectedResults = expectedResukts;
		this.error = "";
		reset();
	}

	@Override
	public void reset() {
		currIdx = 0;		
	}


	@Override
	public boolean atEof() {
		return currIdx>=inputData.length;
	}

	@Override
	public NNetBatch nextBatch(int batchSize) {
		NNetBatch ret = null;
		
		int remaining = inputData.length-currIdx;
		if (remaining==0)
			return null;
		
		int nextbatchSize = (batchSize <= remaining)? batchSize : remaining;
		double[][] nextind = Arrays.copyOfRange(inputData, currIdx, currIdx+nextbatchSize);
		double[][] nextout = Arrays.copyOfRange(expectedResults, currIdx, currIdx+nextbatchSize);
		ret = new NNetBatch(nextbatchSize, new NNetMatrix(nextind),new NNetMatrix(nextout));
		currIdx += nextbatchSize;
		return ret;
	}

	@Override
	public boolean validate() {
		boolean ret = true;
		if (inputData.length!=expectedResults.length)
		{
			error = "Input data  and expected result array lengths to not match";
			ret = false;
		}
		return ret;
	}

	@Override
	public String getError() {
		return error;
	}


}
