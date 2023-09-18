package nnet.matrix.acvt;

public interface ActivationFunction {
	public double f(double in);
	public double grad(double in);
}
