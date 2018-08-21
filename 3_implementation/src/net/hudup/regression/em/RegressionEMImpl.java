package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.defaultExtractVariable;
import static net.hudup.regression.AbstractRegression.defaultExtractVariableName;
import static net.hudup.regression.AbstractRegression.extractNumber;
import static net.hudup.regression.AbstractRegression.findIndex;
import static net.hudup.regression.AbstractRegression.parseIndices;
import static net.hudup.regression.AbstractRegression.solve;

import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.MemFetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.em.ExponentialEM;

/**
 * This class implements default expectation maximization algorithm for regression model in case of missing data, called REM algorithm. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RegressionEMImpl extends ExponentialEM implements RegressionEM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal data.
	 */
	protected LargeStatistics data = null;
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Object[]> xIndices = Util.newList(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Indices for Z data.
	 */
	protected List<Object[]> zIndices = Util.newList(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 * This variable is also used as the indicator of successful learning (not null).
	 */
	protected AttributeList attList = null;
	
	
	/**
	 * Default constructor.
	 */
	public RegressionEMImpl() {
		// TODO Auto-generated constructor stub
		super();
	}
	
	
	@Override
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		Object resulted = null;
		if (prepareInternalData(this.sample))
			resulted = super.learn();
		if (resulted == null)
			clearInternalContent();

		return resulted;
	}


	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		
		if (this.data != null && this.data != this.statistics)
			this.data.clear();
		this.data = null;
	}


	/**
	 * Preparing data.
	 * @param inputSample specified sample.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData(Fetcher<Profile> inputSample) throws Exception {
		clearInternalContent();
		
		this.attList = getSampleAttributeList(inputSample);
		if (this.attList.size() < 2)
			return false;

		//Begin parsing indices
		String cfgIndices = this.getConfig().getAsString(R_INDICES_FIELD);
		if (!parseIndices(cfgIndices, this.attList.size(), this.xIndices, this.zIndices)) //parsing indices
			return false;
		//End parsing indices
		
		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[this.xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (inputSample.next()) {
			Profile profile = inputSample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = extractNumber(extractResponse(profile));
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				double value = extractRegressor(profile, j);
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		inputSample.reset();
		List<Object[]> xIndicesTemp = Util.newList();
		xIndicesTemp.add(this.xIndices.get(0)); //adding -1
		for (int j = 1; j < this.xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(this.xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2)
			return false;
		this.xIndices = xIndicesTemp;
		//End checking existence of values.
		
		//Begin extracting data
		List<double[]> xData = Util.newList();
		List<double[]> zData = Util.newList();
		while (inputSample.next()) {
			Profile profile = inputSample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double[] xVector = new double[this.xIndices.size()]; //1, x1, x2,..., x(n-1)
			double[] zVector = new double[2]; //1, z
			xVector[0] = 1.0;
			zVector[0] = 1.0;
			
			double lastValue = extractNumber(extractResponse(profile));
			if (!Util.isUsed(lastValue))
				zVector[1] = Constants.UNUSED;
			else
				zVector[1] = (double)transformResponse(lastValue, false);
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				double value = extractRegressor(profile, j);
				if (!Util.isUsed(value))
					xVector[j] = Constants.UNUSED;
				else
					xVector[j] = (double)transformRegressor(value, false);
			}
			
			zData.add(zVector);
			xData.add(xVector);
		}
		inputSample.reset();
		//End extracting data
		
		if (xData.size() == 0 || zData.size() == 0)
			return false;
		else {
			this.statistics = this.data = new LargeStatistics(xData, zData);
			return true;
		}
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalContent() {
		this.currentIteration = 0;
		this.currentParameter = this.estimatedParameter = null;
		this.xIndices.clear();
		this.zIndices.clear();
		this.attList = null;
		
		if (this.statistics != null && this.statistics != this.data && (this.statistics instanceof LargeStatistics))
			((LargeStatistics)this.statistics).clear();
		this.statistics = null;
	}
	
	
	@Override
	protected Object expectation(Object currentParameter, Object...info) throws Exception {
		// TODO Auto-generated method stub
		List<Double> alpha = ((ExchangedParameter)currentParameter).getVector();
		List<double[]> betas = ((ExchangedParameter)currentParameter).getMatrix();
		Statistics additionalMean = null;
		if (info != null && info.length > 0 && (info[0] instanceof Statistics))
			additionalMean = (Statistics)info[0];

		int N = this.getData().zData.size();
		List<double[]> zStatistic = Util.newList(N);
		List<double[]> xStatistic = Util.newList(N);
		for (int i = 0; i < N; i++) {
			Statistics stat0 = new Statistics(this.getData().zData.get(i)[1], this.getData().xData.get(i));
			Statistics stat = estimate(stat0, alpha, betas, additionalMean);
			if (stat == null)
				return null;
			
			stat = (stat.checkValid() ? stat : null);
			if (stat == null)
				return null;
			zStatistic.add(new double[] {1.0, stat.getZStatistic()});
			xStatistic.add(stat.getXStatistic());
		}
		
		return new LargeStatistics(xStatistic, zStatistic);
	}

	
	@Override
	protected Object maximization(Object currentStatistic, Object...info) throws Exception {
		// TODO Auto-generated method stub
		LargeStatistics stat = (LargeStatistics)currentStatistic;
		if (stat.isEmpty())
			return null;
		List<double[]> zStatistic = stat.zData;
		List<double[]> xStatistic = stat.xData;
		int N = zStatistic.size();
		ExchangedParameter currentParameter = (ExchangedParameter)getCurrentParameter();
		
		int n = xStatistic.get(0).length; //1, x1, x2,..., x(n-1)
		List<Double> alpha = calcCoeffsByStatistics(xStatistic, zStatistic);
		if (alpha == null) { //If cannot calculate alpha by matrix calculation.
			if (currentParameter != null)
				alpha = DSUtil.toDoubleList(currentParameter.vector); //clone alpha
			else { //Used for initialization so that regression model is always determined.
				alpha = DSUtil.initDoubleList(n, 0.0);
				double alpha0 = 0;
				for (int i = 0; i < N; i++) {
					alpha0 += zStatistic.get(i)[1];
				}
				alpha0 = alpha0 / (double)N; //constant function z = c
				alpha.set(0, alpha0);
			}
		}
		
		List<double[]> betas = Util.newList();
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				double[] beta0 = new double[2];
				beta0[0] = 1;
				beta0[1] = 0;
				betas.add(beta0);
				continue;
			}
			
			List<double[]> Z = Util.newList(N);
			List<Double> x = Util.newList(N);
			for (int i = 0; i < N; i++) {
				Z.add(zStatistic.get(i));
				x.add(xStatistic.get(i)[j]);
			}
			double[] beta = DSUtil.toDoubleArray(calcCoeffs(Z, x));
			if (beta == null) {
				if (currentParameter != null) {
					beta = Arrays.copyOf(currentParameter.matrix.get(j),
						currentParameter.matrix.get(j).length);
				}
				else { //Used for initialization so that regression model is always determined.
					beta = new double[2];
					beta[1] = 0;
					for (int i = 0; i < N; i++)
						beta[0] += xStatistic.get(i)[j];
					beta[0] = beta[0] / (double)N; //constant function x = c
				}
			}
			betas.add(beta);
		}
		
		double coeff = (currentParameter == null ? 1 : currentParameter.getCoeff()); 
		double mean = stat.getZStatisticMean();
		double variance = stat.getZStatisticBiasedVariance();
		
		return new ExchangedParameter(alpha, betas, coeff, mean, variance);
	}
	
	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @param additionalMean mean statistics. This parameter can be null.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	protected Statistics estimate(Statistics stat, List<Double> alpha, List<double[]> betas, Statistics additionalMean) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		//Begin preparing additional means. This code snippet is not important.
		double zAdditionalMean = Constants.UNUSED;
		if (additionalMean != null && Util.isUsed(additionalMean.zStatistic) && !Util.isUsed(zValue))
			zAdditionalMean = additionalMean.zStatistic;
		double[] xAdditionalMean = null;
		if (additionalMean != null && additionalMean.xStatistic != null && additionalMean.xStatistic.length == xVector.length) {
			boolean missing = false;
			for (int i = 0; i < additionalMean.xStatistic.length; i++) {
				if (!Util.isUsed(additionalMean.xStatistic[i])) {
					missing = true;
					break;
				}
			}
			if (!missing) //Additional mean vector must be full.
				xAdditionalMean = additionalMean.xStatistic;
			
			missing = false;
			for (int i = 0; i < xVector.length; i++) {
				if (!Util.isUsed(xVector[i])) {
					missing = true;
					break;
				}
			}
			if (!missing)
				xAdditionalMean = null;
		}
		//End preparing additional means. This code snippet is not important.
		
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else {
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
					if (xAdditionalMean != null)
						xStatistic[j] = (xStatistic[j] + xAdditionalMean[j]) / 2.0;
				}
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		//Estimating missing zi (zStatistic) by equation 7, based on current parameter.
		int a = 0, b = 0, c = 0;
		List<Integer> U = Util.newList(); //Indices of missing values.
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j])) {
				b += alpha.get(j) * xVector[j];
			}
			else {
				a += alpha.get(j) * betas.get(j)[0];
				c += alpha.get(j) * betas.get(j)[1];
				U.add(j);
			}
		}
		if (c != 1) {
			zStatistic = (a + b) / (1 - c);
		}
		else {
			logger.info("Cannot estimate statistic for Z by expectation (#estimate), stop estimating for this statistic here because use of other method is wrong.");
			return null;
		}
		
		if (Util.isUsed(zAdditionalMean))
			zStatistic = (zStatistic + zAdditionalMean) / 2.0; 

		//Estimating missing xij (xStatistic) by equation 5 and estimated zi (zStatistic) above, based on current parameter.
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j]))
				xStatistic[j] = xVector[j];
			else {
				xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
				if (xAdditionalMean != null)
					xStatistic[j] = (xStatistic[j] + xAdditionalMean[j]) / 2.0;
			}
		}
		
		//Balance process is removed with support of mixture model and so list U is not used.
		return new Statistics(zStatistic, xStatistic);
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		int N = this.getData().zData.size();
		int n = this.getData().xData.get(0).length;
		
		List<Double> alpha0 = DSUtil.initDoubleList(n, 0.0);
		List<double[]> betas0 = Util.newList(n);
		for (int j = 0; j < n; j++) {
			double[] beta0 = new double[2];
			if (j == 0) {
				beta0[0] = 1;
				beta0[1] = 0;
			}
			else {
				beta0[0] = 0;
				beta0[1] = 0;
			}
			betas0.add(beta0);
		}
		ExchangedParameter parameter0 = new ExchangedParameter(alpha0, betas0);
		
		List<double[]> xStatistic = Util.newList();
		List<double[]> zStatistic = Util.newList();
		for (int i = 0; i < N; i++) {
			double[] zVector = this.getData().zData.get(i);
			if (!Util.isUsed(zVector[1]))
				continue;
			
			double[] xVector = this.getData().xData.get(i);
			boolean missing = false;
			for (int j = 0; j < xVector.length; j++) {
				if (!Util.isUsed(xVector[j])) {
					missing = true;
					break;
				}
			}
			
			if (!missing) {
				xStatistic.add(xVector);
				zStatistic.add(zVector);
			}
		}
		
		if (xStatistic.size() == 0 || zStatistic.size() == 0)
			return parameter0;
		
		LargeStatistics data = new LargeStatistics(xStatistic, zStatistic);
		try {
			ExchangedParameter parameter = (ExchangedParameter) maximization(data);
			return (parameter != null ? parameter : parameter0); 
		}
		catch (Throwable e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return parameter0;
	}

	
	@Override
	protected boolean terminatedCondition(Object currentParameter, Object estimatedParameter, Object... info) {
		// TODO Auto-generated method stub
		ExchangedParameter parameter1 = ((ExchangedParameter)currentParameter);
		ExchangedParameter parameter2 = ((ExchangedParameter)estimatedParameter);
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		
		return parameter1.terminatedCondition(parameter2, threshold);
	}

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.estimatedParameter == null)
			return null;
		List<Double> alpha = ((ExchangedParameter)this.estimatedParameter).getVector();
		if (alpha == null || alpha.size() == 0)
			return null;
		
		if (input == null || !(input instanceof Profile))
			return null; //only support profile input currently
		Profile profile = (Profile)input;
		
		double sum = alpha.get(0);
		for (int j = 0; j < alpha.size() - 1; j++) {
			double value = extractRegressor(profile, j + 1); //due to x = (1, x1, x2,..., xn-1) and so index 0 indicates value 1.
			if (!Util.isUsed(value))
				return null;
			sum += alpha.get(j + 1) * (double)transformRegressor(value, false); 
		}
		
		return transformResponse(sum, true);
	}
	
	
	/**
	 * Getting internal data. Actually, this method returns the current statistics.
	 * @return internal data which is the current statistics.
	 */
	protected LargeStatistics getData() {
		return this.data;
	}
	
	
	/**
	 * Testing whether missing values are fulfilled.
	 * @return true if missing values are fulfilled.
	 */
	public boolean isMissingDataFilled() {
		return (this.statistics != null && this.statistics != this.data);
	}
	
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "rem";
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		RegressionEMImpl em = new RegressionEMImpl();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		return em;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_FIELD_DEFAULT);
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.getParameter() == null)
			return "";
		ExchangedParameter exParameter = ((ExchangedParameter)this.getParameter());
		List<Double> alpha = exParameter.getVector();
		if (alpha.size() == 0)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append(transformResponse(extractResponseName(), false) + " = " + MathUtil.format(alpha.get(0)));
		for (int j = 0; j < alpha.size() - 1; j++) {
			double coeff = alpha.get(j + 1);
			String regressorExpr = "(" + transformRegressor(extractRegressorName(j + 1), false).toString() + ")";
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + regressorExpr);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + regressorExpr);
		}
		
		double c = exParameter.getCoeff();
		double mean = exParameter.getMean();
		double variance = exParameter.getVariance();
		if (Util.isUsed(c) || Util.isUsed(mean) || Util.isUsed(variance))
			buffer.append(": ");
		
		if (Util.isUsed(c))
			buffer.append("coeff=" + MathUtil.format(c));
		if (Util.isUsed(mean))
			buffer.append(", mean=" + MathUtil.format(mean));
		if (Util.isUsed(variance))
			buffer.append(", variance=" + MathUtil.format(variance));

		return buffer.toString();
	}


	@Override
	public String parameterToShownText(Object parameter, Object...info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof ExchangedParameter))
			return "";
		
		ExchangedParameter exParameter = ((ExchangedParameter)parameter);
		List<Double> alpha = exParameter.getVector();
		StringBuffer buffer = new StringBuffer();
		for (int j = 0; j < alpha.size(); j++) {
			if (j > 0)
				buffer.append(", ");
			buffer.append(MathUtil.format(alpha.get(j)));
		}
		
		double c = exParameter.getCoeff();
		double mean = exParameter.getMean();
		double variance = exParameter.getVariance();
		if (Util.isUsed(c) || Util.isUsed(mean) || Util.isUsed(variance))
			buffer.append(": ");
		
		if (Util.isUsed(c))
			buffer.append("coeff=" + c);
		if (Util.isUsed(mean))
			buffer.append(", mean=" + mean);
		if (Util.isUsed(variance))
			buffer.append(", variance=" + variance);
			
		return buffer.toString();
	}

	
	/**
	 * Calculating coefficients based on regressors X (statistic X) and response variable Z (statistic Z).
	 * Both statistic X and statistic Z contain 1 at first column.
	 * @param xStatistic regressors X (statistic X).
	 * @param zStatistic response variable Z (statistic Z).
	 * @return coefficients based on regressors X (statistic X) and response variable Z (statistic Z). Return null if any error raises.
	 */
	protected static List<Double> calcCoeffsByStatistics(List<double[]> xStatistic, List<double[]> zStatistic) {
		List<Double> z = Util.newList(zStatistic.size());
		for (int i = 0; i < zStatistic.size(); i++) {
			z.add(zStatistic.get(i)[1]);
		}
		
		return calcCoeffs(xStatistic, z);
	}
	
	
	/**
	 * Calculating coefficients based on data matrix and data vector.
	 * This method will be improved in the next version.
	 * @param X specified data matrix.
	 * @param z specified data vector.
	 * @return coefficients base on data matrix and data vector. Return null if any error raises.
	 */
	protected static List<Double> calcCoeffs(List<double[]> X, List<Double> z) {
		int N = z.size();
		int n = X.get(0).length;
		
		List<double[]> A = Util.newList(n);
		for (int i = 0; i < n; i++) {
			double[] aRow = new double[n];
			A.add(aRow);
			for (int j = 0; j < n; j++) {
				double sum = 0;
				for (int k = 0; k < N; k++) {
					sum += X.get(k)[i] * X.get(k)[j];
				}
				aRow[j] = sum;
			}
		}
		
		List<Double> b = Util.newList(n);
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int k = 0; k < N; k++)
				sum += X.get(k)[i] * z.get(k);
			
			b.add(sum);
		}
		
		return solve(A, b);
	}
	
	
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param profile specified profile.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in internal indices.
	 * So index 0 always indicates to value 1. 
	 * @return value of regressor (X) extracted from specified profile.
	 */
	protected double extractRegressor(Profile profile, int index) {
		// TODO Auto-generated method stub
		return defaultExtractVariable(profile, xIndices, index);
	}


	/**
	 * Extracting name of regressor (X).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in internal indices.
	 * So index 0 always indicates to value &apos;#noname&apos;. 
	 * @return text of regressor (X) extracted.
	 */
	protected String extractRegressorName(int index) {
		// TODO Auto-generated method stub
		return defaultExtractVariableName(attList, xIndices, index);
	}

	
	/**
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 */
	@Override
	public Object extractResponse(Profile profile) {
		// TODO Auto-generated method stub
		return defaultExtractVariable(profile, zIndices, 1);
	}


	/**
	 * Extracting name of response variable (Z).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @return text of response variable (Z) extracted.
	 */
	protected String extractResponseName() {
		// TODO Auto-generated method stub
		return defaultExtractVariableName(attList, zIndices, 1);
	}


	/**
	 * Transforming independent variable X.
	 * In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param x specified variable X.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of X.
	 */
	protected Object transformRegressor(Object x, boolean inverse) {
		// TODO Auto-generated method stub
		return x;
	}


	/**
	 * Transforming independent variable Z.
	 * In the most general case that each index is an mathematical expression, this method is not focused but is useful in some cases.
	 * @param z specified variable Z.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of Z.
	 */
	protected Object transformResponse(Object z, boolean inverse) {
		// TODO Auto-generated method stub
		return z;
	}


	/**
	 * Calculating residual mean from specified sample.
	 * @param inputSample specified sample.
	 * @return residual mean from specified sample.
	 */
	public double residualMean(Fetcher<Profile> inputSample) {
		double ss = 0;
		int count = 0;
		
		try {
			while (inputSample.next()) {
				Profile profile = inputSample.pick();
				if (profile == null)
					continue;
				
				double zValue = extractNumber(extractResponse(profile));
				double executedValue = extractNumber(execute(profile));
				if (Util.isUsed(zValue) && Util.isUsed(executedValue)) {
					double d = executedValue - zValue;
					ss += d*d;
					count++;
				}
			}
			inputSample.reset();
			
			if (count == 0)
				return Constants.UNUSED;
			else
				return ss / (double)count;
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		return Constants.UNUSED;
	}
	
	
	/**
	 * Estimating given sample.
	 * @param inputSample given sample.
	 * @param additionalMean additional mean.
	 * @return estimated sample.
	 */
	public Fetcher<Profile> estimate(Fetcher<Profile> inputSample, Statistics additionalMean) {
		// TODO Auto-generated method stub
		ExchangedParameter parameter = (ExchangedParameter)this.getParameter();
		if (parameter == null)
			return null;
		
		List<Double> alpha = parameter.getVector();
		List<double[]> betas = parameter.getMatrix();
		RegressionEMImpl em = new RegressionEMImpl();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		try {
			if (!em.prepareInternalData(inputSample))
				return null;
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		
		int N = em.getData().zData.size();
		List<double[]> zStatistic = Util.newList(N);
		List<double[]> xStatistic = Util.newList(N);
		for (int i = 0; i < N; i++) {
			Statistics stat0 = new Statistics(em.getData().zData.get(i)[1], em.getData().xData.get(i));
			Statistics stat = this.estimate(stat0, alpha, betas, additionalMean);
			if (stat == null)
				return null;
			
			stat = (stat.checkValid() ? stat : null);
			if (stat == null)
				return null;
			zStatistic.add(new double[] {1.0, stat.getZStatistic()});
			xStatistic.add(stat.getXStatistic());
		}
		
		AttributeList attRef = getSampleAttributeList(inputSample);
		List<Profile> profiles = Util.newList();
		for (int i = 0; i < N; i++) {
			Profile profile = new Profile(attRef);
			double[] xvector = xStatistic.get(i);
			double z = zStatistic.get(i)[1];
			profile.setValue(attRef.size() - 1, z);
			
			for (int j = 0; j < attRef.size() - 1; j++) {
				int foundX = findIndex(em.xIndices, j);
				if (foundX >= 0)
					profile.setValue(j, xvector[foundX]);
			}
			profiles.add(profile);
		}
		em.clearInternalContent();

		return new MemFetcher<>(profiles);
	}


}
