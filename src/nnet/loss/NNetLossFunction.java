package nnet.loss;

import nnet.matrix.Matrix;

public interface NNetLossFunction {
	public static NNetLossFunction SQUARED_ERROR = new NNetLossFuncSqrdError();
	public static NNetLossFunction MEAN_SQRD_ERROR = new NNetLossFuncMeanSquaredError();
	public static NNetLossFunction ABSOLUTE_ERROR = new NNetLossFuncAbsoluteError();
	
	public double getLoss(Matrix actual, Matrix expected);
}
