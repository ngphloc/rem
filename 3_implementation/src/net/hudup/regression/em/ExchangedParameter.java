package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.notSatisfy;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Logger;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;

/**
 * This class represents the exchanged parameter for this REM algorithm.
 * @author Loc Nguyen
 * @version 1.0
 */
public class ExchangedParameter {

	
	/**
	 * Logger of this class.
	 */
	protected final static Logger logger = Logger.getLogger(ExchangedParameter.class);

	
	/**
	 * Alpha coefficients for Z statistics.
	 */
	protected List<Double> alpha = null;
	
	
	/**
	 * Beta coefficients for X statistics as matrix.
	 * Each row of the matrix a a list of two beta coefficients for a regressor.
	 * As a convent, regression 1 also has a list of two beta coefficients.
	 */
	protected List<double[]> betas = null; 
	
	
	/**
	 * Additional information for Z statistics.
	 */
	protected ExchangedParameterInfo zInfo = null;
	
	
	/**
	 * Default constructor.
	 */
	private ExchangedParameter() {
		
	}
	
	
	/**
	 * Constructor with specified alpha and betas.
	 * @param alpha specified alpha. It must be not null.
	 * @param betas specified betas. It must be not null.
	 */
	public ExchangedParameter(List<Double> alpha, List<double[]> betas) {
		this.alpha = alpha;
		this.betas = betas;
		this.zInfo = new ExchangedParameterInfo();
	}

	
	/**
	 * Constructor with specified alpha, betas, and Z additional information.
	 * @param alpha specified alpha. It must be not null.
	 * @param betas specified betas. It must be not null.
	 * @param zInfo Additional information for Z statistic.
	 */
	public ExchangedParameter(List<Double> alpha, List<double[]> betas, ExchangedParameterInfo zInfo) {
		this.alpha = alpha;
		this.betas = betas;
		this.zInfo = zInfo;
	}
	
	
	@Override
	public Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		ExchangedParameter newParameter = new ExchangedParameter();
		newParameter.alpha = (this.alpha != null ? DSUtil.toDoubleList(this.alpha) : null);
		
		if (this.betas != null) {
			newParameter.betas = Util.newList(this.betas.size());
			for (double[] array : this.betas) {
				newParameter.betas.add(Arrays.copyOf(array, array.length));
			}
		}
		
		newParameter.zInfo = (ExchangedParameterInfo)this.zInfo.clone();
		return newParameter;
	}

	
	/**
	 * Getting alpha parameter.
	 * @return alpha parameter.
	 */
	public List<Double> getAlpha() {
		return alpha;
	}
	
	
	/**
	 * Setting alpha parameter.
	 * @param alpha specified parameter.
	 */
	public void setAlpha(List<Double> alpha) {
		this.alpha = alpha;
	}

	
	/**
	 * Getting betas.
	 * @return betas.
	 */
	public List<double[]> getBetas() {
		return betas;
	}
	
	
	/**
	 * Setting betas parameter.
	 * @param betas specified betas.
	 */
	public void setBetas(List<double[]> betas) {
		this.betas = betas;
	}

	/**
	 * Getting Z additional information.
	 * @return Z additional information.
	 */
	public ExchangedParameterInfo getZInfo() {
		return zInfo;
	}
	
	
	/**
	 * Setting Z additional information.
	 * @param zInfo specified Z additional information.
	 */
	public void setZInfo(ExchangedParameterInfo zInfo) {
		this.zInfo = zInfo;
	}
	
	
	@Override
	public String toString() {
		// TODO Auto-generated method stub
		StringBuffer buffer = new StringBuffer();
		if (this.alpha != null) {
			for (int j = 0; j < this.alpha.size(); j++) {
				if (j > 0)
					buffer.append(", ");
				buffer.append(MathUtil.format(this.alpha.get(j)));
			}
		}
		
		if (this.zInfo != null) {
			if (buffer.length() > 0)
				buffer.append(": ");
			buffer.append(this.zInfo.toString());
		}
		
		return buffer.toString();
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
		List<Double> alpha1 = previousParameter != null ? previousParameter.getAlpha() : null;
		List<Double> alpha2 = currentParameter.getAlpha();
		List<Double> alpha3 = this.getAlpha();
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
		List<double[]> betas1 = previousParameter != null ? previousParameter.getBetas() : null;
		List<double[]> betas2 = currentParameter.getBetas();
		List<double[]> betas3 = this.getBetas();
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

		if (this.zInfo == null)
			return true;
		else
			return this.zInfo.terminatedCondition(
				threshold,
				currentParameter.zInfo,
				previousParameter != null ? previousParameter.zInfo : null);
	}

	
	/**
	 * Calculating the normal condition probabilities of the specified parameters given response value (Z).
	 * @param z given response value (Z).
	 * @param parameters arrays of parameters.
	 * @return condition probabilities of the specified parameters given response value (Z).
	 */
	public static double[] normalZCondProbs(double z, ExchangedParameter...parameters) {
		double[] coeffs = new double[parameters.length];
		double[] means = new double[parameters.length];
		double[] variances = new double[parameters.length];
		for (int i = 0; i < parameters.length; i++) {
			coeffs[i] = parameters[i].zInfo.getCoeff();
			means[i] = parameters[i].zInfo.getMean();
			variances[i] = parameters[i].zInfo.getVariance();
		}

		return ExchangedParameterInfo.normalCondProbs(z, coeffs, means, variances);
	}

	
	
	/**
	 * Calculating the scalar product of specified coefficients and X variable (regressor).
	 * @param alpha specified coefficients
	 * @param xVector specified X variable (regressor).
	 * @return the scalar product of specified coefficients and X variable (regressor).
	 */
	public static double product(double[] alpha, double[] xVector) {
		double mean = 0;
		for (int i = 0; i < alpha.length; i++) {
			mean += alpha[i] * xVector[i];
		}
		return mean;
	}
	

	/**
	 * This class represents additional information to learn parameter.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class ExchangedParameterInfo {

		/**
		 * Logger of this class.
		 */
		protected final static Logger logger = Logger.getLogger(ExchangedParameterInfo.class);

		/**
		 * Probability associated with this component.
		 */
		protected double coeff = 1;
		
		/**
		 * Mean associated with this component.
		 */
		protected double mean = 0;
		
		/**
		 * Variance associated with this component.
		 */
		protected double variance = 1;
		
		/**
		 * Indicator to learning requirement.
		 */
		protected boolean requiredLearning = true;
		
		/**
		 * Default constructor.
		 */
		public ExchangedParameterInfo() {
			
		}
		
		/**
		 * Constructor with specified component probability, mean, and variance.
		 * @param coeff specified coefficient.
		 * @param mean specified mean.
		 * @param variance specified variance.
		 */
		public ExchangedParameterInfo(double coeff, double mean, double variance) {
			this.coeff = coeff;
			this.mean = mean;
			this.variance = variance;
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
		 * Setting associated variance.
		 * @param mean specified variance.
		 */
		public void setVariance(double variance) {
			this.variance = variance;
		}

		/**
		 * Getting indicator of learning requirement.
		 * @return true if learning is required.
		 */
		public boolean isRequiredLearning() {
			return requiredLearning;
		}
		
		/**
		 * Setting indicator of learning requirement.
		 * @param requiredLearning specified indicator of learning requirement.
		 */
		public void setRequiredLearning(boolean requiredLearning) {
			this.requiredLearning = requiredLearning;
		}
		
		/**
		 * Learning this additional information from large statistics.
		 * @param stat large statistics.
		 */
		public void learn(LargeStatistics stat) {
			this.mean = stat.getZStatisticMean();
			this.variance = stat.getZStatisticBiasedVariance();
		}
		
		@Override
		public Object clone() throws CloneNotSupportedException {
			// TODO Auto-generated method stub
			ExchangedParameterInfo newInfo = new ExchangedParameterInfo(this.coeff, this.mean, this.variance);
			newInfo.requiredLearning = this.requiredLearning;
			return newInfo;
		}

		/**
		 * Testing the terminated condition between this additional information (estimated additional information) and other additional information (current additional information).
		 * @param threshold specified threshold
		 * @param currentInfo other specified additional information (current additional information).
		 * @param previousInfo previous additional information is used to avoid skip-steps in optimization for too acute function.
		 * It also solve the over-fitting problem. Please pay attention to it.
		 * @return true if the terminated condition is satisfied.
		 */
		public boolean terminatedCondition(double threshold, ExchangedParameterInfo currentInfo, ExchangedParameterInfo previousInfo) {
			//Testing coefficient
			double c1 = previousInfo != null ? previousInfo.getCoeff() : Constants.UNUSED;
			double c2 = currentInfo.getCoeff();
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
			double mean1 = previousInfo != null ? previousInfo.getMean() : Constants.UNUSED;
			double mean2 = currentInfo.getMean();
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
			double variance1 = previousInfo != null ? previousInfo.getVariance() : Constants.UNUSED;
			double variance2 = currentInfo.getVariance();
			double variance3 = this.getVariance();
			if (Util.isUsed(variance2) && Util.isUsed(variance3)) {
				if (notSatisfy(variance3, variance2, threshold)) {
					if (!Util.isUsed(variance1))
						return false;
					else if (notSatisfy(variance3, variance1, threshold)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
						return false;
				}
			}
			else if (Util.isUsed(variance3) || Util.isUsed(variance2))
				return false;

			return true;
			
		}
		
		@Override
		public String toString() {
			// TODO Auto-generated method stub
			StringBuffer buffer = new StringBuffer();
			buffer.append("coeff=" + MathUtil.format(coeff));
			buffer.append(", mean=" + MathUtil.format(mean));
			buffer.append(", variance=" + MathUtil.format(variance));
			
			return buffer.toString();
		}

		
		/**
		 * Calculating the condition probabilities given value, mean, and variance.
		 * @param value given value.
		 * @param coeffs array of coefficients.
		 * @param means given means.
		 * @param variances given variances.
		 * @return condition probabilities given value, means, and variances. 
		 */
		public static double[] normalCondProbs(double value, double[] coeffs, double[] means, double[] variances) {
			double[] numerators = new double[coeffs.length];
			double denominator = 0;
			
			for (int i = 0; i < coeffs.length; i++) {
				double p = normalPDF(value, means[i], variances[i]);
				double product = coeffs[i] * p;
				
				denominator += product;
				numerators[i] = product;
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
		 * @param value specified response value z.
		 * @param mean specified mean.
		 * @param variance specified variance.
		 * @return value evaluated from the normal probability density function.
		 */
		private static double normalPDF(double value, double mean, double variance) {
			double d = value - mean;
			if (variance == 0)
				variance = variance + Double.MIN_VALUE; //Solving the problem of zero variance.
			return (1.0 / Math.sqrt(2*Math.PI*variance)) * Math.exp(-(d*d) / (2*variance));
		}
		
	}


}




