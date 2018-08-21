package net.hudup.regression.em;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Logger;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.AbstractAlg;
import net.hudup.core.logistic.DSUtil;

/**
 * This class represents the exchanged parameter for this REM algorithm.
 * @author Loc Nguyen
 * @version 1.0
 */
public class ExchangedParameter {

	
	/**
	 * Logger of this class.
	 */
	protected final static Logger logger = Logger.getLogger(AbstractAlg.class);

	
	/**
	 * Vector parameter. As usual, it is alpha coefficients for Z statistics.
	 */
	protected List<Double> vector = null;
	
	
	/**
	 * Matrix parameter represents beta coefficients for X statistics.
	 * Each row of the matrix a a list of two beta coefficients for a regressor.
	 * As a convent, regression 1 also has a list of two beta coefficients.
	 */
	protected List<double[]> matrix = null; 
	
	
	/**
	 * Probability associated with this component. This variable is not used for normal regression model.
	 */
	protected double coeff = 1;
	
	
	/**
	 * Mean associated with this component. This variable is not used for normal regression model.
	 */
	protected double mean = 0;
	
	
	/**
	 * Variance associated with this component. This variable is not used for normal regression model.
	 */
	protected double variance = 1;
	
	
	/**
	 * Default constructor.
	 */
	private ExchangedParameter() {
		
	}
	
	
	/**
	 * Constructor with specified vector and matrix.
	 * @param vector specified vector. It must be not null but can be zero-length.
	 * @param matrix specified matrix. It must be not null but can be zero-length.
	 */
	public ExchangedParameter(List<Double> vector, List<double[]> matrix) {
		this.vector = vector;
		this.matrix = matrix;
	}
	
	
	/**
	 * Constructor with specified vector, matrix, component probability, mean, and variance.
	 * @param vector specified vector. It must be not null but can be zero-length.
	 * @param matrix specified matrix. It must be not null but can be zero-length.
	 * @param coeff specified coefficient.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 */
	public ExchangedParameter(List<Double> vector, List<double[]> matrix, double coeff, double mean, double variance) {
		this.vector = vector;
		this.matrix = matrix;
		this.coeff = coeff;
		this.mean = mean;
		this.variance = variance;
	}

	
	@Override
	public Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		ExchangedParameter newParameter = new ExchangedParameter();
		newParameter.vector = (this.vector != null ? DSUtil.toDoubleList(this.vector) : null);
		
		if (this.matrix != null) {
			newParameter.matrix = Util.newList(this.matrix.size());
			for (double[] array : this.matrix) {
				newParameter.matrix.add(Arrays.copyOf(array, array.length));
			}
		}
		
		newParameter.coeff = this.coeff;
		newParameter.mean = this.mean;
		newParameter.variance = this.variance;
		
		return newParameter;
	}

	/**
	 * Getting vector parameter.
	 * @return vector parameter.
	 */
	public List<Double> getVector() {
		return vector;
	}
	
	
	/**
	 * Setting vector parameter.
	 * @param vector specified parameter.
	 */
	public void setVector(List<Double> vector) {
		this.vector = vector;
	}

	
	/**
	 * Getting matrix.
	 * @return matrix.
	 */
	public List<double[]> getMatrix() {
		return matrix;
	}
	
	
	/**
	 * Setting matrix parameter.
	 * @param matrix specified parameter.
	 */
	public void setMatrix(List<double[]> matrix) {
		this.matrix = matrix;
	}

	
	/**
	 * Getting coefficient.
	 * @return coefficient.
	 */
	public double getCoeff() {
		return coeff;
	}
	
	
	/**
	 * Setting coefficient.
	 * @param coeff specified coefficient.
	 */
	public void setCoeff(double coeff) {
		this.coeff = coeff;
	}

	
	/**
	 * Getting associated probability of component.
	 * @return associated probability of component.
	 */
	public double getMean() {
		return mean;
	}
	
	
	/**
	 * Setting associated mean.
	 * @param mean specified mean.
	 */
	public void setMean(double mean) {
		this.mean = mean;
	}

	
	/**
	 * Getting associated variance of component.
	 * @return associated variance of component.
	 */
	public double getVariance() {
		return variance;
	}
	
	
	/**
	 * Testing whether the deviation between estimated value and current value is not satisfied a threshold.
	 * @param estimatedValue estimated value.
	 * @param currentValue current value.
	 * @param threshold specified threshold.
	 * @return true if the deviation between estimated value and current value is not satisfied a threshold.
	 */
	private boolean notSatisfy(double estimatedValue, double currentValue, double threshold) {
		return Math.abs(estimatedValue - currentValue) > threshold * Math.abs(currentValue);
	}
	
	
	/**
	 * Testing the terminated condition between this parameter (estimated parameter) and other parameter (current parameter).
	 * @param threshold specified threshold
	 * @param currentParameter other specified parameter (current parameter).
	 * @param previousParameter previous parameter is used to avoid skip-steps in optimization for too acute function.
	 * It also solve the over-fitting problem. Please pay attention to it.
	 * @return true if the terminated condition is satisfied.
	 */
	public boolean terminatedCondition(double threshold, ExchangedParameter currentParameter, ExchangedParameter previousParameter) {
		// TODO Auto-generated method stub
		List<Double> alpha1 = previousParameter != null ? previousParameter.getVector() : null;
		List<Double> alpha2 = currentParameter.getVector();
		List<Double> alpha3 = this.getVector();
		if (alpha3 != null && alpha2 != null) {
			for (int i = 0; i < alpha2.size(); i++) {
				if (notSatisfy(alpha3.get(i), alpha2.get(i), threshold)) {
					if (alpha1 == null)
						return false;
					else if (notSatisfy(alpha3.get(i), alpha1.get(i), threshold)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
						return false;
				}
			}
		}
		else if(alpha3 != null || alpha2 != null)
			return false;
		
		//It is possible not to test beta coefficients
		List<double[]> betas1 = previousParameter != null ? previousParameter.getMatrix() : null;
		List<double[]> betas2 = currentParameter.getMatrix();
		List<double[]> betas3 = this.getMatrix();
		if(betas3 != null && betas2 != null) {
			for (int i = 0; i < betas2.size(); i++) {
				double[]  beta1 = betas1 != null ? betas1.get(i) : null;  
				double[]  beta2 = betas2.get(i);
				double[]  beta3 = betas3.get(i);
				
				for (int j = 0; j < beta2.length; j++) {
					if (notSatisfy(beta3[j], beta2[j], threshold)) {
						if (beta1 == null)
							return false;
						else if (notSatisfy(beta3[j], beta1[j], threshold)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
							return false;
					}
				}
			}
		}
		else if(betas3 != null || betas2 != null)
			return false;
		//It is possible not to test beta coefficients

		//Testing coefficient
		double c1 = previousParameter != null ? previousParameter.getCoeff() : Constants.UNUSED;
		double c2 = currentParameter.getCoeff();
		double c3 = this.getCoeff();
		if (Util.isUsed(c2) && Util.isUsed(c3)) {
			if (notSatisfy(c3, c2, threshold)) {
				if (!Util.isUsed(c1))
					return false;
				else if (notSatisfy(c3, c1, threshold)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
					return false;
			}
		}
		else if (Util.isUsed(c3) || Util.isUsed(c2))
			return false;
		
		//Testing mean
		double mean1 = previousParameter != null ? previousParameter.getMean() : Constants.UNUSED;
		double mean2 = currentParameter.getMean();
		double mean3 = this.getMean();
		if (Util.isUsed(mean2) && Util.isUsed(mean3)) {
			if (notSatisfy(mean3, mean2, threshold)) {
				if (!Util.isUsed(mean1))
					return false;
				else if (notSatisfy(mean3, mean1, threshold)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
					return false;
			}
		}
		else if (Util.isUsed(mean3) || Util.isUsed(mean2))
			return false;

		//Testing variance
		double variance1 = previousParameter != null ? previousParameter.getVariance() : Constants.UNUSED;
		double variance2 = currentParameter.getVariance();
		double variance3 = this.getVariance();
		if (Util.isUsed(variance2) && Util.isUsed(variance3)) {
			if (notSatisfy(variance3, variance2, threshold)) {
				if (!Util.isUsed(variance1))
					return false;
				else if (notSatisfy(variance3, variance1, threshold))
					return false;
			}
		}
		else if (Util.isUsed(variance3) || Util.isUsed(variance2))
			return false;

		return true;
	}

	
	/**
	 * Setting associated variance.
	 * @param mean specified variance.
	 */
	public void setVariance(double variance) {
		this.variance = variance;
	}


	/**
	 * Getting coefficients from specified list of parameters.
	 * @param parameters specified list of parameters.
	 * @return coefficients from specified list of parameters.
	 */
	public static double[] getCoeffs(ExchangedParameter...parameters) {
		double[] coeffs = new double[parameters.length];
		for (int i = 0; i < parameters.length; i++) {
			coeffs[i] = parameters[i].getCoeff();
		}
		
		return coeffs;
	}
	
	
	/**
	 * Getting means from specified list of parameters.
	 * @param parameters specified list of parameters.
	 * @return means from specified list of parameters.
	 */
	public static double[] getMeans(ExchangedParameter...parameters) {
		double[] means = new double[parameters.length];
		for (int i = 0; i < parameters.length; i++) {
			means[i] = parameters[i].getMean();
		}
		
		return means;
	}

	
	/**
	 * Getting variances from specified list of parameters.
	 * @param parameters specified list of parameters.
	 * @return variances from specified list of parameters.
	 */
	public static double[] getVariances(ExchangedParameter...parameters) {
		double[] variances = new double[parameters.length];
		for (int i = 0; i < parameters.length; i++) {
			variances[i] = parameters[i].getVariance();
		}
		
		return variances;
	}

	
	/**
	 * Calculating the normal condition probabilities of the specified parameters given response value (Z).
	 * @param z given response value (Z).
	 * @param parameters arrays of parameters.
	 * @return condition probabilities of the specified parameters given response value (Z).
	 */
	public static double[] normalCondProbs(double z, ExchangedParameter...parameters) {
		double[] coeffs = getCoeffs(parameters); 
		double[] means = getMeans(parameters);
		double[] variances = getVariances(parameters);
		
		return normalCondProbs(z, coeffs, means, variances);
	}

	
	/**
	 * Calculating the condition probabilities given response value (Z), mean, and variance.
	 * @param z given response value (Z).
	 * @param coeffs array of coefficients.
	 * @param means given means.
	 * @param variances given variances.
	 * @return condition probabilities given response value (Z), means, and variances. 
	 */
	private static double[] normalCondProbs(double z, double[] coeffs, double[] means, double[] variances) {
		double[] numerators = new double[coeffs.length];
		double denominator = 0;
		
		for (int i = 0; i < coeffs.length; i++) {
			double p = normalPDF(z, means[i], variances[i]);
			double value = coeffs[i] * p;
			
			denominator += value;
			numerators[i] = value;
		}
		
		double[] condProbs = new double[coeffs.length];
		for (int i = 0; i < coeffs.length; i++) {
			if (denominator == 0) {
				condProbs[i] = 1.0 / coeffs.length;
				logger.warn("Reset uniform conditional probability of component due to zero denominator");
			}
			else
				condProbs[i] = numerators[i] / denominator;
		}
		
		return condProbs; 
	}

	
	/**
	 * Evaluating the normal probability density function with specified mean and variance.
	 * Inherited class can re-defined this density function.
	 * @param z specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the normal probability density function.
	 */
	private static double normalPDF(double z, double mean, double variance) {
		double d = z - mean;
		if (variance == 0)
			variance = variance + Double.MIN_VALUE; //Solving the problem of zero variance.
		return (1.0 / Math.sqrt(2*Math.PI*variance)) * Math.exp(-(d*d) / (2*variance));
	}
	
	
	/**
	 * Calculating the mean of specified coefficients and X variable (regressor).
	 * @param alpha specified coefficients
	 * @param xVector specified X variable (regressor).
	 * @return the mean of specified coefficients and X variable (regressor).
	 */
	public static double mean(double[] alpha, double[] xVector) {
		double mean = 0;
		for (int i = 0; i < alpha.length; i++) {
			mean += alpha[i] * xVector[i];
		}
		return mean;
	}
	
	
}
