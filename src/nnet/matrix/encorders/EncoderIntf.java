package nnet.matrix.encorders;

import nnet.matrix.NNetMatrix;

public interface EncoderIntf {
	public void fit(NNetMatrix m);
	public void fit(NNetMatrix m,int... columns);
	
	public NNetMatrix transform(NNetMatrix m);
	public NNetMatrix transform(NNetMatrix m,int... columns);
	public NNetMatrix transformInPlace(NNetMatrix m);
	public NNetMatrix transformInPlace(NNetMatrix m,int... columns);

	public NNetMatrix fitAndTransform(NNetMatrix m);
	public NNetMatrix fitAndTransform(NNetMatrix m,int... columns);
	public NNetMatrix fitAndTransformInPlace(NNetMatrix m);
	public NNetMatrix fitAndTransformInPlace(NNetMatrix m,int... columns);
	
	public boolean isFitted();
	public double[] getVariance();
	public double[] getMean();
	public double[] getStdv();
	
}
