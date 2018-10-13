package net.hudup.regression.em;

import static net.hudup.regression.AbstractRM.notSatisfy;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.log4j.Logger;

import net.hudup.core.Cloneable;
import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.regression.LargeStatistics;

/**
 * This class represents the exchanged parameter for the REM algorithm.
 * @author Loc Nguyen
 * @version 1.0
 */
public class ExchangedParameter implements Cloneable {

	
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
	 * Probability associated with this component.
	 */
	protected double coeff = Constants.UNUSED;
	

	/**
	 * Variance associated with this component.
	 */
	protected double zVariance = Constants.UNUSED;

	
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
	}

	
	/**
	 * Constructor with specified alpha, betas, coefficient, and Z variance.
	 * @param alpha specified alpha. It must be not null.
	 * @param betas specified betas. It must be not null.
	 * @param coeff specified coefficient.
	 * @param zVariance specified Z variance.
	 */
	public ExchangedParameter(List<Double> alpha, List<double[]> betas, double coeff, double zVariance) {
		this.alpha = alpha;
		this.betas = betas;
		this.coeff = coeff;
		this.zVariance = zVariance;
	}
	
	
	@Override
	public Object clone() {
		// TODO Auto-generated method stub
		ExchangedParameter newParameter = new ExchangedParameter();
		newParameter.coeff = this.coeff;
		newParameter.alpha = (this.alpha != null ? DSUtil.toDoubleList(this.alpha) : null);
		
		if (this.betas != null) {
			newParameter.betas = Util.newList(this.betas.size());
			for (double[] array : this.betas) {
				newParameter.betas.add(Arrays.copyOf(array, array.length));
			}
		}
		
		newParameter.zVariance = this.zVariance;
		
		return newParameter;
	}

	
	/**
	 * Getting coefficient.
	 * @return coefficient.
	 */
	public double getCoeff() {
		return this.coeff;
	}
	
	
	/**
	 * Setting coefficient.
	 * @param coeff specified coefficient.
	 */
	public void setCoeff(double coeff) {
		this.coeff = coeff;
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
	 * Getting associated Z variance of component.
	 * @return associated Z variance of component.
	 */
	public double getZVariance() {
		return this.zVariance;
	}

	
	/**
	 * Setting associated Z variance.
	 * @param mean specified Z variance.
	 */
	public void setZVariance(double zVariance) {
		this.zVariance = zVariance;
	}
	
	
	/**
	 * Estimating variance by large statistics.
	 * @param stats large statistics.
	 * @return estimated variance
	 */
	public double estimateZVariance(LargeStatistics stats) {
		List<double[]> xData = stats.getXData();
		List<double[]> zData = stats.getZData();
		
		double ss = 0;
		int N = 0;
		for (int i = 0; i < xData.size(); i++) {
			double[] xVector = xData.get(i);
			double zValue = zData.get(i)[1];
			double zEstimatedValue = mean(xVector);
			
			if (Util.isUsed(zValue) && Util.isUsed(zEstimatedValue)) {
				ss += (zEstimatedValue - zValue) * (zEstimatedValue - zValue);
				N++;
			}
		}
		return ss / N;
	}

	
	/**
	 * Estimating mean of response variable Z by regressors.
	 * @param xData value of regressors.
	 * @return estimated mean.
	 */
	public double estimateZMean(List<double[]> xData) {
		double sum = 0;
		int N = 0;
		for (int i = 0; i < xData.size(); i++) {
			double[] xVector = xData.get(i);
			double zEstimatedValue = mean(xVector);
			
			if (Util.isUsed(zEstimatedValue)) {
				sum += zEstimatedValue;
				N++;
			}
		}
		
		return sum / N;
	}

	
	/**
	 * Testing the terminated condition between this parameter (estimated parameter) and other parameter (current parameter).
	 * Only internal alpha coefficients and beta coefficients are tested. 
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
		
		double coeff1 = previousParameter != null ? previousParameter.coeff : Constants.UNUSED;
		double coeff2 = currentParameter.coeff;
		double coeff3 = this.coeff;
		if (Util.isUsed(coeff3) && Util.isUsed(coeff2)) {
			if (notSatisfy(coeff3, coeff2, threshold)) {
				if (!Util.isUsed(coeff1))
					return false;
				else if (notSatisfy(coeff3, coeff1, threshold)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
					return false;
			}
		}
		else if (Util.isUsed(coeff3) || Util.isUsed(coeff2))
			return false;
		
		double zVariance1 = previousParameter != null ? previousParameter.zVariance : Constants.UNUSED;
		double zVariance2 = currentParameter.zVariance;
		double zVariance3 = this.zVariance;
		if (Util.isUsed(zVariance3) && Util.isUsed(zVariance2)) {
			if (notSatisfy(zVariance3, zVariance2, threshold)) {
				if (!Util.isUsed(zVariance1))
					return false;
				else if (notSatisfy(zVariance3, zVariance1, threshold)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
					return false;
			}
		}
		else if (Util.isUsed(zVariance3) || Util.isUsed(zVariance2))
			return false;

		return true;
	}

	
	/**
	 * Testing whether the alpha coefficient of other parameter equals the alpha coefficient of other parameter.
	 * @param other other parameter.
	 * @return true the alpha coefficient of other parameter equals the alpha coefficient of other parameter.
	 */
	public boolean alphaEquals(ExchangedParameter other) {
		if (other == null)
			return false;
		if (this.alpha == null && other.alpha == null)
			return true;
		else if (this.alpha == null || other.alpha == null)
			return false;
		else if (this.alpha.size() != other.alpha.size())
			return false;
		 
		for (int j = 0; j < this.alpha.size(); j++) {
			if (this.alpha.get(j) != other.alpha.get(j))
				return false;
 		}
		
		return true;
	}
	
	
	/**
	 * Testing whether all alpha coefficients are zero.
	 * @return true if all alpha coefficients are zero.
	 */
	public boolean isNullAlpha() {
		if (this.alpha == null || this.alpha.size() == 0)
			return true;
		
		for (int j = 0; j < this.alpha.size(); j++) {
			if (this.alpha.get(j) != 0)
				return false;
 		}
		return true;
	}
	
	
	@Override
	public String toString() {
		// TODO Auto-generated method stub
		if (this.alpha == null)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		for (int j = 0; j < this.alpha.size(); j++) {
			if (j > 0)
				buffer.append(", ");
			buffer.append(MathUtil.format(this.alpha.get(j)));
		}
		
		buffer.append(": ");
		buffer.append("coeff=" + MathUtil.format(this.coeff));
		buffer.append(", z-variance=" + MathUtil.format(this.zVariance));
		
		return buffer.toString();
	}


	/**
	 * Calculating the scalar product of internal coefficients and X variable (regressor).
	 * @param xVector specified X variable (regressor).
	 * @return the scalar product of specified coefficients and X variable (regressor).
	 */
	public double mean(double[] xVector) {
		return mean(this.alpha, xVector);
	}

	
	/**
	 * Calculating the scalar product of specified coefficients and X variable (regressor).
	 * @param alpha specified coefficients
	 * @param xVector specified X variable (regressor).
	 * @return the scalar product of specified coefficients and X variable (regressor).
	 */
	public static double mean(List<Double> alpha, double[] xVector) {
		double mean = 0;
		if (xVector.length < alpha.size()) {
			double[] xNewVector = new double[alpha.size()];
			Arrays.fill(xNewVector, 1);
			int start = alpha.size() - xVector.length;
			for (int i = start; i < alpha.size(); i++) {
				xNewVector[start] = xVector[i - start];
			}
			xVector = xNewVector;
		}
		
		for (int i = 0; i < alpha.size(); i++) {
			if (Util.isUsed(alpha.get(i)) && Util.isUsed(xVector[i]))
				mean += alpha.get(i) * xVector[i];
			else
				return Constants.UNUSED;
		}
		return mean;
	}

	
	/**
	 * Evaluating the normal probability density function with specified mean and variance.
	 * Inherited class can re-defined this density function.
	 * @param value specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the normal probability density function.
	 */
	public static double normalPDF(double value, double mean, double variance) {
		double d = value - mean;
		if (variance == 0)
			variance = variance + Double.MIN_VALUE; //Solving the problem of zero variance.
		return (1.0 / Math.sqrt(2*Math.PI*variance)) * Math.exp(-(d*d) / (2*variance));
	}

	
	/**
	 * Calculating the normal condition probabilities of the specified parameters given regressor values (X) and response value Z.
	 * Inherited class can re-define this method. In current version, only normal probability density function is used.
	 * @param parameterList list of specified parameters.
	 * @param xData given regressor values (X).
	 * @param zData response values (Z).
	 * @return condition probabilities of the specified parameters given regressor values (X) and response value Z.
	 */
	public static List<Double> normalZCondProbs(List<ExchangedParameter> parameterList, List<double[]> xData, List<double[]> zData) {
		if (parameterList == null || xData == null || parameterList.size() == 0 || xData.size() == 0 || zData.size() == 0)
			return Util.newList();
		
		List<Double> condProbs = Util.newList(parameterList.size());
		List<Double> numerators = Util.newList(parameterList.size());
		double denominator = 0;
		for (int i = 0; i < parameterList.size(); i++) {
			double[] xVector = xData.size() == parameterList.size() ? xData.get(i) : xData.get(0);
			double zValue = zData.size() == parameterList.size() ? zData.get(i)[1] : zData.get(0)[1];
			
			double coeff = parameterList.get(i).coeff;
			double zMean = parameterList.get(i).mean(xVector);
			double zVariance = parameterList.get(i).zVariance;
			
			double p = normalPDF(zValue, zMean, zVariance);
			double product = coeff * p;
			
			denominator += product;
			numerators.add(product);
		}
		
		for (int i = 0; i < parameterList.size(); i++) {
			if (denominator == 0) {
				condProbs.add(1.0 / (double)parameterList.size());
				logger.warn("Reset uniform conditional probability of component due to zero denominator");
			}
			else
				condProbs.add(numerators.get(i) / denominator);
		}
		
		return condProbs;
	}

	
	/**
	 * Calculating the normal condition probabilities of the specified parameters given regressor values (X) and response value Z.
	 * @param parameterList list of specified parameters.
	 * @param xVector given regressor values (X).
	 * @param zVector response values (Z).
	 * @return condition probabilities of the specified parameters given regressor values (X) and response value Z.
	 */
	public static List<Double> normalZCondProbs(List<ExchangedParameter> parameterList, double[] xVector, double[] zVector) {
		return normalZCondProbs(parameterList, Arrays.asList(xVector), Arrays.asList(zVector));
	}
	
	
	/**
	 * Calculating the normal probabilities of the specified parameters given regressor values (X) and response value Z.
	 * @param parameterList list of specified parameters.
	 * @param xData given regressor values (X).
	 * @param zData response values (Z).
	 * @param vicinity this parameter is depreciated.
	 * @return condition probabilities of the specified parameters given regressor values (X) and response value Z.
	 */
	public static List<Double> normalZPDF(List<ExchangedParameter> parameterList, List<double[]> xData, List<double[]> zData, double vicinity) {
		if (parameterList == null || xData == null || parameterList.size() == 0 || xData.size() == 0 || zData.size() == 0)
			return Util.newList();
		
		List<Double> condProbs = Util.newList(parameterList.size());
		for (int i = 0; i < parameterList.size(); i++) {
			double[] xVector = xData.size() == parameterList.size() ? xData.get(i) : xData.get(0);
			double zValue = zData.size() == parameterList.size() ? zData.get(i)[1] : zData.get(0)[1];
			
			double zMean = parameterList.get(i).mean(xVector);
			double zVariance = parameterList.get(i).zVariance;
			
//			double p1 = normalPDF(zValue - vicinity, zMean, zVariance);
//			double p2 = normalPDF(zValue + vicinity, zMean, zVariance);
//			condProbs.add(p2 - p1);
			
			double p = normalPDF(zValue, zMean, zVariance);
			condProbs.add(p);
		}
		
		return condProbs;
	}


	/**
	 * Calculating the normal probabilities of the specified parameters given regressor values (X) and response value Z.
	 * @param parameterList list of specified parameters.
	 * @param xVector given regressor values (X).
	 * @param zVector response values (Z).
	 * @return condition probabilities of the specified parameters given regressor values (X) and response value Z.
	 */
	public static List<Double> normalZPDF(List<ExchangedParameter> parameterList, double[] xVector, double[] zVector) {
		return normalZPDF(parameterList, Arrays.asList(xVector), Arrays.asList(zVector), 0);
	}
	
	
	/**
	 * Cloning the specified collection of parameters.
	 * @param collection specified collection of parameters.
	 * @return cloned list of parameters.
	 */
	public static List<ExchangedParameter> clone(Collection<ExchangedParameter> collection) {
		List<ExchangedParameter> list = Util.newList(collection.size());
		for (ExchangedParameter parameter : collection) {
			list.add((ExchangedParameter)parameter.clone());
		}
		
		return list;
	}
	
	
}


