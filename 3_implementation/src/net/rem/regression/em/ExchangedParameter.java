/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.em;

import static net.rem.regression.RMAbstract.notSatisfy;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;

import net.hudup.core.Cloneable;
import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.parser.TextParserUtil;
import net.rem.regression.LargeStatistics;
import net.rem.regression.RMAbstract;
import net.rem.regression.Statistics;

/**
 * This class represents the exchanged parameter for the REM algorithm.
 * @author Loc Nguyen
 * @version 1.0
 */
public class ExchangedParameter implements Cloneable, Serializable {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;


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
	 * Parameter of normal distribution of X variable, excluding the first 1 value.
	 * Note, X statistics is (1, x1, x2,..., xn) but normal distribution of X variable has x1, x2,..., xn. 
	 */
	protected NormalDisParameter xNormalDisParameter = null;
	
	
	/**
	 * Default constructor.
	 */
	private ExchangedParameter() {
		
	}
	
	
	/**
	 * Constructor with specified alpha.
	 * @param alpha specified alpha. It must be not null.
	 */
	public ExchangedParameter(List<Double> alpha) {
		this.alpha = alpha;
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
		this(alpha, betas, coeff, zVariance, null);
	}
	
	
	/**
	 * Constructor with specified alpha, betas, coefficient, Z variance, and parameter of normal distribution of X variable.
	 * @param alpha specified alpha. It must be not null.
	 * @param betas specified betas. It must be not null.
	 * @param coeff specified coefficient.
	 * @param zVariance specified Z variance.
	 * @param xNormalDisParameter parameter of normal distribution of X variable, excluding the first 1 value.
	 * Note, X statistics is (1, x1, x2,..., xn) but normal distribution of X variable has x1, x2,..., xn.
	 */
	public ExchangedParameter(List<Double> alpha, List<double[]> betas, double coeff, double zVariance, NormalDisParameter xNormalDisParameter) {
		this.alpha = alpha;
		this.betas = betas;
		this.coeff = coeff;
		this.zVariance = zVariance;
		this.xNormalDisParameter = xNormalDisParameter;
	}

	
	@Override
	public Object clone() {
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
		
		if (this.xNormalDisParameter != null)
			newParameter.xNormalDisParameter = (NormalDisParameter)this.xNormalDisParameter.clone(); 
		
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
	 * @param zVariance specified Z variance.
	 */
	public void setZVariance(double zVariance) {
		this.zVariance = zVariance;
	}
	
	
	/**
	 * Getting parameter of normal distribution of X variable.
	 * @return parameter of normal distribution of X variable.
	 */
	public NormalDisParameter getXNormalDisParameter() {
		return xNormalDisParameter;
	}
	
	
	/**
	 * Setting parameter of normal distribution of X variable.
	 * @param xNormalDisParameter parameter of normal distribution of X variable.
	 */
	public void setXNormalDisParameter(NormalDisParameter xNormalDisParameter) {
		this.xNormalDisParameter = xNormalDisParameter;
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
	 * Calculating likelihood of specified statistics.
	 * @param stats specified large statistics.
	 * @param log logarithm flag.
	 * @return likelihood of specified statistics.
	 */
	public double likelihood(LargeStatistics stats, boolean log) {
		return likelihood(stats, Constants.UNUSED, log);
	}
	
	
	/**
	 * Calculating likelihood of specified statistics.
	 * @param stats specified large statistics.
	 * @param variance specified variance.
	 * @param log logarithm flag.
	 * @return likelihood of specified statistics.
	 */
	public double likelihood(LargeStatistics stats, double variance, boolean log) {
		if (stats == null) return Constants.UNUSED;
		int n = stats.size();
		if (n == 0) return Constants.UNUSED;
		
		variance = Util.isUsed(variance) ? variance : estimateZVariance(stats);
		variance = Util.isUsed(variance) ? variance : 1;
		double lh = 1;
		for (int i = 0; i < n; i++) {
			try {
				Statistics stat = stats.getStatistic(i);
				if (stat == null) continue;

				double mean = mean(stat.getXStatistic());
				double prob = normalPDF(stat.getZStatistic(), mean, variance);
				if (log)
					lh += prob > 0 ? Math.log(prob) : 0;
				else
					lh *= prob;
			} catch (Throwable e) {LogUtil.trace(e);}
		}
		
		return lh;
	}

	
	/**
	 * Testing the terminated condition between this parameter (estimated parameter) and other parameter (current parameter).
	 * This method only tests alpha coefficients and mixture coefficients (mixture weights). 
	 * @param threshold specified threshold
	 * @param currentParameter other specified parameter (current parameter).
	 * @param previousParameter previous parameter is used to avoid skip-steps in optimization for too acute function.
	 * @param ratioMode flag to indicate whether the threshold is for ratio.
	 * It also solve the over-fitting problem. Please pay attention to it.
	 * @return true if the terminated condition is satisfied.
	 */
	public boolean terminatedCondition(double threshold, ExchangedParameter currentParameter, ExchangedParameter previousParameter, boolean ratioMode) {
		List<Double> alpha1 = previousParameter != null ? previousParameter.getAlpha() : null;
		List<Double> alpha2 = currentParameter.getAlpha();
		List<Double> alpha3 = this.getAlpha();
		if (alpha3 != null && alpha2 != null) {
			for (int i = 0; i < alpha2.size(); i++) {
				if (notSatisfy(alpha3.get(i), alpha2.get(i), threshold, ratioMode)) {
					if (alpha1 == null)
						return false;
					else if (notSatisfy(alpha3.get(i), alpha1.get(i), threshold, ratioMode)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
						return false;
				}
			}
		}
		else if(alpha3 != null || alpha2 != null)
			return false;
		
		return true;
	}
	
	
	/**
	 * Testing the terminated condition between this parameter (estimated parameter) and other parameter (current parameter).
	 * This method tests all sub-parameters and so it is currently not used. It is used for backup.
	 * @param threshold specified threshold
	 * @param currentParameter other specified parameter (current parameter).
	 * @param previousParameter previous parameter is used to avoid skip-steps in optimization for too acute function.
	 * @param ratioMode flag to indicate whether the threshold is for ratio.
	 * It also solve the over-fitting problem. Please pay attention to it.
	 * @return true if the terminated condition is satisfied.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private boolean terminatedCondition0(double threshold, ExchangedParameter currentParameter, ExchangedParameter previousParameter, boolean ratioMode) {
		List<Double> alpha1 = previousParameter != null ? previousParameter.getAlpha() : null;
		List<Double> alpha2 = currentParameter.getAlpha();
		List<Double> alpha3 = this.getAlpha();
		if (alpha3 != null && alpha2 != null) {
			for (int i = 0; i < alpha2.size(); i++) {
				if (notSatisfy(alpha3.get(i), alpha2.get(i), threshold, ratioMode)) {
					if (alpha1 == null)
						return false;
					else if (notSatisfy(alpha3.get(i), alpha1.get(i), threshold, ratioMode)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
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
					if (notSatisfy(beta3[j], beta2[j], threshold, ratioMode)) {
						if (beta1 == null)
							return false;
						else if (notSatisfy(beta3[j], beta1[j], threshold, ratioMode)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
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
			if (notSatisfy(coeff3, coeff2, threshold, ratioMode)) {
				if (!Util.isUsed(coeff1))
					return false;
				else if (notSatisfy(coeff3, coeff1, threshold, ratioMode)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
					return false;
			}
		}
		else if (Util.isUsed(coeff3) || Util.isUsed(coeff2))
			return false;
		
		double zVariance1 = previousParameter != null ? previousParameter.zVariance : Constants.UNUSED;
		double zVariance2 = currentParameter.zVariance;
		double zVariance3 = this.zVariance;
		if (Util.isUsed(zVariance3) && Util.isUsed(zVariance2)) {
			if (notSatisfy(zVariance3, zVariance2, threshold, ratioMode)) {
				if (!Util.isUsed(zVariance1))
					return false;
				else if (notSatisfy(zVariance3, zVariance1, threshold, ratioMode)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
					return false;
			}
		}
		else if (Util.isUsed(zVariance3) || Util.isUsed(zVariance2))
			return false;

		NormalDisParameter xNormalDisParameter1 = previousParameter != null ? previousParameter.getXNormalDisParameter() : null;
		NormalDisParameter xNormalDisParameter2 = currentParameter.getXNormalDisParameter();
		NormalDisParameter xNormalDisParameter3 = this.getXNormalDisParameter();
		if(xNormalDisParameter3 != null && xNormalDisParameter2 != null) {
			if (!xNormalDisParameter3.terminatedCondition(threshold, xNormalDisParameter2, null, ratioMode)) {
				if (xNormalDisParameter1 == null)
					return false;
				else if (!xNormalDisParameter3.terminatedCondition(threshold, xNormalDisParameter1, null, ratioMode))
					return false;
			}
		}
		else if(xNormalDisParameter3 != null || xNormalDisParameter2 != null)
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
		if (this.alpha == null)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		for (int j = 0; j < this.alpha.size(); j++) {
			if (j > 0) buffer.append(", ");
			buffer.append(MathUtil.format(this.alpha.get(j)));
		}
		
		buffer.append(": ");
		buffer.append("coeff=" + MathUtil.format(this.coeff));
		buffer.append(", z-variance=" + MathUtil.format(this.zVariance));
		
		if (xNormalDisParameter != null)
			buffer.append(", x-parameter=(" + xNormalDisParameter.toString() + ")");
			
		return buffer.toString();
	}


	/**
	 * Calculating the scalar product of internal coefficients and X variable (regressor).
	 * @param xVector specified X variable (regressor), xVector[0] = 1 always.
	 * @return the scalar product of specified coefficients and X variable (regressor).
	 */
	public double mean(double[] xVector) {
		return mean(this.alpha, xVector);
	}

	
	/**
	 * Calculating the scalar product of specified coefficients and X variable (regressor).
	 * @param alpha specified coefficients.
	 * @param xVector specified X variable (regressor), xVector[0] = 1 always.
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
	 * Evaluating the normal probability density function with specified mean and variance given multivariate data.
	 * Inherited class can re-defined this density function.
	 * @param value specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the normal probability density function given multivariate data.
	 */
	public static double normalPDF(List<Double> value, List<Double> mean, List<double[]> variance) {
		int n = mean.size();
		
		double det = RMAbstract.matrixDeterminant(variance);
		if (det == 0) {
			boolean equal = true;
			for (int i = 0; i < n; i++) {
				if (value.get(i) != mean.get(i)) {
					equal = false;
					break;
				}
			}
			
			if (equal)
				return normalPDF(0, 0, 0);
			else
				return 0;
		}
		double v1 = Math.sqrt(Math.pow(2*Math.PI, n)*det);

		List<Double> d = DSUtil.initDoubleList(n, 0);
		for (int i = 0; i < n; i++) {
			d.add(value.get(i) - mean.get(i));
		}
		
		List<double[]> inverseVariance = RMAbstract.matrixInverse(variance);
		double v2 = 0;
		if (inverseVariance != null && inverseVariance.size() > 0) {
			for (int j = 0; j < n; j++) {
				double sum = 0;
				for (int i = 0; i < n; i++) {
					sum += d.get(i)*inverseVariance.get(i)[j];
				}
				v2 += sum*d.get(j);
			}
		}
		
		return (1.0 / v1) * Math.exp(-v2/2.0);
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
		if (variance == 0 && mean != value) return 0;
		if (variance == 0 && mean == value) return 1;
		
//		variance = variance != 0 ? variance : Float.MIN_VALUE;
		double d = value - mean;
		return (1.0 / (Math.sqrt(2*Math.PI*variance))) * Math.exp(-(d*d) / (2*variance));
	}

	
	/**
	 * Evaluating the normal cumulative density function with specified mean and variance.
	 * Inherited class can re-defined this density function.
	 * @param value specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the normal probability density function.
	 */
	public static double normalCDF(double value, double mean, double variance) {
		if (variance == 0 && mean != value) return 0;
		if (variance == 0 && mean == value) return 1;
//		variance = variance != 0 ? variance : Float.MIN_VALUE;

		return new NormalDistribution(mean, Math.sqrt(variance)).cumulativeProbability(value);
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
			if (denominator != 0 && Util.isUsed(denominator))
				condProbs.add(numerators.get(i) / denominator);
			else {
				condProbs.add(1.0 / (double)parameterList.size());
				LogUtil.warn("Reset uniform conditional probability of component due to zero denominator");
			}
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
	 * @param vicinity this parameter is depreciated. It is now not used.
	 * @return condition probabilities of the specified parameters given regressor values (X) and response value Z.
	 */
	private static List<Double> normalZPDF(List<ExchangedParameter> parameterList, List<double[]> xData, List<double[]> zData, double vicinity) {
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
	
	
	/**
	 * This class represents parameter of multivariate normal distribution.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class NormalDisParameter implements Cloneable, Serializable {
		
		/**
		 * Default serial version UID.
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Mean.
		 */
		private List<Double> mean = null;
		
		/**
		 * Variance.
		 */
		private List<double[]> variance = null;
		
		/**
		 * Default constructor.
		 */
		private NormalDisParameter() {
			
		}
		
		/**
		 * Constructor of specified mean and variance.
		 * @param mean specified mean.
		 * @param variance specified variance.
		 */
		public NormalDisParameter(List<Double> mean, List<double[]> variance) {
			this.mean = mean;
			this.variance = variance;
		}
		
		/**
		 * Constructor with a large statistics.
		 * @param stat given a large statistics.
		 */
		public NormalDisParameter(LargeStatistics stat) {
			if (stat == null) return;
			List<double[]> xData = stat.getXData();
			if (xData == null || xData.size() == 0) return;
			
			int n = xData.get(0).length - 1;
			if (n <= 0) return;
			
			List<Double> xMean = DSUtil.initDoubleList(n, 0);
			int N = xData.size();
			for (int i = 0; i < N; i++) {
				double[] x = xData.get(i);
				for (int j = 0; j < n; j++) {
					xMean.set(j, xMean.get(j) + x[j+1]);
				}
			}
			for (int j = 0; j < n; j++) {
				xMean.set(j, xMean.get(j) / (double)N);
			}
			
			
			List<double[]> xVariance = Util.newList(n);
			for (int i = 0; i < n; i++) {
				double[] x = new double[n];
				Arrays.fill(x, 0);
				xVariance.add(x);
			}
			
			for (int i = 0; i < N; i++) {
				double[] d = xData.get(i);
				for (int j = 0; j < n; j++) {d[j+1] = d[j+1] - xMean.get(j);}
				
				for (int j = 0; j < n; j++) {
					double[] x = xVariance.get(j);
					for (int k = 0; k < n; k++) {
						x[k] = x[k] + d[j+1]*d[k+1];
					}
				}
			}
			
			for (int j = 0; j < n; j++) {
				double[] x = xVariance.get(j);
				for (int k = 0; k < n; k++) {
					x[k] = x[k] / (double)N;
				}
			}
			
			
//			if (!RMAbstract.matrixIsInvertible(xVariance))
//				xVariance = createDiagonalVariance(n, 1);
//			if (xVariance.size() > 0) xVariance.get(0)[0] = Double.MIN_VALUE;

			this.mean = xMean;
			this.variance = xVariance;
		}
		
		/**
		 * Constructor with a large statistics and conditional probabilities.
		 * @param stat given a large statistics.
		 * @param kCondProbs conditional probabilities.
		 */
		public NormalDisParameter(LargeStatistics stat, List<Double> kCondProbs) {
			if (stat == null) return;
			List<double[]> xData = stat.getXData();
			if (xData == null || xData.size() == 0) return;
			
			int n = xData.get(0).length - 1;
			if (n <= 0) return;
			
			int N = xData.size();
			double sumCondProbs = 0;
			for (int i = 0; i < N; i++) {sumCondProbs += kCondProbs.get(i);}
			
			List<Double> xMean = DSUtil.initDoubleList(n, 0);
			for (int i = 0; i < N; i++) {
				double[] x = xData.get(i);
				for (int j = 0; j < n; j++) {
					if (sumCondProbs != 0)
						xMean.set(j, xMean.get(j) + kCondProbs.get(i)*x[j+1]);
					else
						xMean.set(j, xMean.get(j) + x[j+1]);
				}
			}
			for (int j = 0; j < n; j++) {
				if (sumCondProbs != 0)
					xMean.set(j, xMean.get(j)/sumCondProbs);
				else
					xMean.set(j, xMean.get(j)/N);
			}
			
			
			List<double[]> xVariance = Util.newList(n);
			for (int i = 0; i < n; i++) {
				double[] x = new double[n];
				Arrays.fill(x, 0);
				xVariance.add(x);
			}
			
			for (int i = 0; i < N; i++) {
				double[] d = xData.get(i);
				for (int j = 0; j < n; j++) {d[j+1] = d[j+1] - xMean.get(j);}
				
				for (int j = 0; j < n; j++) {
					double[] x = xVariance.get(j);
					for (int k = 0; k < n; k++) {
						if (sumCondProbs != 0)
							x[k] = x[k] + kCondProbs.get(i)*d[j+1]*d[k+1];
						else
							x[k] = x[k] + d[j+1]*d[k+1];
					}
				}
			}
			
			for (int j = 0; j < n; j++) {
				double[] x = xVariance.get(j);
				for (int k = 0; k < n; k++) {
					if (sumCondProbs != 0)
						x[k] = x[k]/sumCondProbs;
					else
						x[k] = x[k]/N;
				}
			}
			
			
//			if (!RMAbstract.matrixIsInvertible(xVariance))
//				xVariance = createDiagonalVariance(n, 1);
//			if (xVariance.size() > 0) xVariance.get(0)[0] = Double.MIN_VALUE;
			
			this.mean = xMean;
			this.variance = xVariance;
		}

		/**
		 * Getting mean.
		 * @return mean.
		 */
		public List<Double> getMean() {
			return mean;
		}
		
		/**
		 * Getting variance.
		 * @return variance.
		 */
		public List<double[]> getVariance() {
			return variance;
		}

		/**
		 * Testing the terminated condition between this parameter (estimated parameter) and other parameter (current parameter).
		 * @param threshold specified threshold
		 * @param currentParameter other specified parameter (current parameter).
		 * @param previousParameter previous parameter is used to avoid skip-steps in optimization for too acute function.
		 * @param ratioMode flag to indicate whether the threshold is for ratio.
		 * It also solve the over-fitting problem. Please pay attention to it.
		 * @return true if the terminated condition is satisfied.
		 */
		public boolean terminatedCondition(double threshold, NormalDisParameter currentParameter, NormalDisParameter previousParameter, boolean ratioMode) {
			List<Double> mean1 = previousParameter != null ? previousParameter.getMean() : null;
			List<Double> mean2 = currentParameter.getMean();
			List<Double> mean3 = this.getMean();
			if (mean3 != null && mean2 != null) {
				for (int i = 0; i < mean2.size(); i++) {
					if (notSatisfy(mean3.get(i), mean2.get(i), threshold, ratioMode)) {
						if (mean1 == null)
							return false;
						else if (notSatisfy(mean3.get(i), mean1.get(i), threshold, ratioMode)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
							return false;
					}
				}
			}
			else if(mean3 != null || mean2 != null)
				return false;
			
			List<double[]> variance1 = previousParameter != null ? previousParameter.getVariance() : null;
			List<double[]> variance2 = currentParameter.getVariance();
			List<double[]> variance3 = this.getVariance();
			if(variance3 != null && variance2 != null) {
				for (int i = 0; i < variance2.size(); i++) {
					double[]  v1 = variance1 != null ? variance1.get(i) : null;  
					double[]  v2 = variance2.get(i);
					double[]  v3 = variance3.get(i);
					
					for (int j = 0; j < v2.length; j++) {
						if (notSatisfy(v3[j], v2[j], threshold, ratioMode)) {
							if (v1 == null)
								return false;
							else if (notSatisfy(v3[j], v1[j], threshold, ratioMode)) //previous parameter is used to avoid skip-steps in optimization for too acute function.
								return false;
						}
					}
				}
			}
			else if(variance3 != null || variance2 != null)
				return false;
			
			return true;
		}

		@Override
		public Object clone() {
			NormalDisParameter newParameter = new NormalDisParameter();
			newParameter.mean = (this.mean != null ? DSUtil.toDoubleList(this.mean) : null);
			
			if (this.variance != null) {
				newParameter.variance = Util.newList(this.variance.size());
				for (double[] array : this.variance) {
					newParameter.variance.add(Arrays.copyOf(array, array.length));
				}
			}
			
			return newParameter;
		}

		@Override
		public String toString() {
			if (mean == null || mean.size() == 0 || variance == null || variance.size() == 0)
				return "";
			
			StringBuffer buffer = new StringBuffer();
//			buffer.append("mean=(" + DSUtil.shortenVerbalName(TextParserUtil.toTextFormatted(mean, ",")) + "), ");
			buffer.append("mean=(" + TextParserUtil.toTextFormatted(mean, ",") + "), ");
			buffer.append("variance=(");
			StringBuffer var = new StringBuffer();
			for (int i = 0; i < variance.size(); i++) {
				if (i > 0) var.append(", ");
				var.append(TextParserUtil.toTextFormatted(variance.get(i), ","));
			}
//			buffer.append(DSUtil.shortenVerbalName(var.toString()) + ")");
			buffer.append(var.toString() + ")");
			
			return buffer.toString();
		}
		
		
		/**
		 * Creating diagonal co-variance matrix with specified value.
		 * @param n dimension of the returned diagonal co-variance matrix.
		 * @param value specified value.
		 * @return diagonal co-variance matrix with specified value.
		 */
		public static List<double[]> createDiagonalVariance(int n, double value) {
			List<double[]> matrix = Util.newList(n);
			for (int i = 0; i < n; i++) {
				double[] row = new double[n];
				Arrays.fill(row, 0);
				matrix.add(row);
				
				row[i] = value;
			}
			
			return matrix;
		}
	}

	
}


