package nnet.loss;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.matrix.NNetMatrix;

public interface NNetLossFunction {
	public static NNetLossFunction SQUARED_ERROR = new NNetLossFuncSqrdError();
	public static NNetLossFunction MEAN_SQRD_ERROR = new NNetLossFuncMeanSquaredError();
	public static NNetLossFunction ABSOLUTE_ERROR = new NNetLossFuncAbsoluteError();
	
	public double getLoss(NNetMatrix actual, NNetMatrix expected) throws MxlibInvalidMatrixOp;
}
