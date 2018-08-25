package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.createProfile;
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
	 * Name of variance calculation field.
	 */
	public final static String R_CALC_VARIANCE_FIELD = "r_calc_variance";

	
	/**
	 * Default value variance calculation field.
	 */
	public final static boolean R_CALC_VARIANCE_DEFAULT = false;

	
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
		List<Double> alpha = ((ExchangedParameter)currentParameter).getAlpha();
		List<double[]> betas = ((ExchangedParameter)currentParameter).getBetas();
		LargeStatistics data = null;
		if (info != null && info.length > 0 && (info[0] instanceof LargeStatistics))
			data = (LargeStatistics)info[0];
		else
			data = this.data;
		
		int N = data.getZData().size();
		List<double[]> zStatistic = Util.newList(N);
		List<double[]> xStatistic = Util.newList(N);
		for (int i = 0; i < N; i++) {
			Statistics stat0 = new Statistics(data.getZData().get(i)[1], data.getXData().get(i));
			Statistics stat = estimate(stat0, alpha, betas);
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
		List<double[]> zStatistic = stat.getZData();
		List<double[]> xStatistic = stat.getXData();
		int N = zStatistic.size();
		ExchangedParameter currentParameter = (ExchangedParameter)getCurrentParameter();
		
		int n = xStatistic.get(0).length; //1, x1, x2,..., x(n-1)
		List<Double> alpha = calcCoeffsByStatistics(xStatistic, zStatistic);
		if (alpha == null || alpha.size() == 0) { //If cannot calculate alpha by matrix calculation.
			if (currentParameter != null)
				alpha = DSUtil.toDoubleList(currentParameter.getAlpha()); //clone alpha
			else { //Used for initialization so that regression model is always determined.
				alpha = DSUtil.initDoubleList(n, 0.0);
				double alpha0 = 0;
				for (int i = 0; i < N; i++)
					alpha0 += zStatistic.get(i)[1];
				alpha.set(0, alpha0 / (double)N); //constant function z = c
			}
		}
		
		List<double[]> betas = Util.newList(n);
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
			List<Double> beta = calcCoeffs(Z, x);
			if (beta == null || beta.size() == 0) {
				if (currentParameter != null)
					beta = DSUtil.toDoubleList(currentParameter.getBetas().get(j));
				else { //Used for initialization so that regression model is always determined.
					beta = DSUtil.initDoubleList(2, 0);
					double beta0 = 0;
					for (int i = 0; i < N; i++)
						beta0 += xStatistic.get(i)[j];
					beta.set(0, beta0 / (double)N); //constant function x = c
				}
			}
			betas.add(DSUtil.toDoubleArray(beta));
		}
		
		ExchangedParameter newParameter = new ExchangedParameter(alpha, betas);
		if (getConfig().getAsBoolean(R_CALC_VARIANCE_FIELD))
			newParameter.setZVariance(newParameter.estimateZVariance(stat));
		else {
			if (currentParameter == null)
				newParameter.setZVariance(Constants.UNUSED);
			else if (Util.isUsed(currentParameter.getZVariance()))
				newParameter.setZVariance(newParameter.estimateZVariance(stat));
			else
				newParameter.setZVariance(Constants.UNUSED);
		}
		
		return newParameter;
	}
	
	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * Balance process is removed because it is over-fitting or not stable. Balance process is the best in some cases.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	protected Statistics estimate(Statistics stat, List<Double> alpha, List<double[]> betas) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		//Estimating missing zi (zStatistic) by equation 7, based on current parameter.
		double a = 0, b = 0, c = 0;
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
			zStatistic = (a + b) / (1.0 - c);
		}
		else {
			logger.info("Cannot estimate statistic for Z by expectation (#estimate), stop estimating for this statistic here because use of other method is wrong.");
			return null;
		}
		
		//Estimating missing xij (xStatistic) by equation 5 and estimated zi (zStatistic) above, based on current parameter.
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j]))
				xStatistic[j] = xVector[j];
			else
				xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
		}
		
		//Balance process is removed because it is over-fitting or not stable. Balance process is the best in some cases. So list U is not used.
		return new Statistics(zStatistic, xStatistic);
	}

	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * Balance process is removed because it is over-fitting or not stable. Balance process is the best in some cases.
	 * This method is better than {@link #estimate(Statistics, List, List, Statistics)} method but it is not stable for long regression model having many regressors
	 * because solving a set of many equations can cause approximate solution or non-solution problem.   
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @return estimated statistics with specified parameters alpha and beta. Return null if any error raises.
	 */
	@Deprecated
	protected Statistics estimateInverse(Statistics stat, List<Double> alpha, List<double[]> betas) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		List<Integer> U = Util.newList();
		double b = 0;
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j])) {
				b += alpha.get(j) * xVector[j];
				xStatistic[j] = xVector[j]; //existent xij
			}
			else
				U.add(j);
		}

		if (U.size() > 0) {
			//Estimating missing xij (xStatistic) by equation 8, based on current parameter.
			List<double[]> A = Util.newList(U.size());
			List<Double> y = Util.newList(U.size());
			
			for (int i = 0; i < U.size(); i++) {
				double[] aRow = new double[U.size()];
				A.add(aRow);
				for (int j = 0; j < U.size(); j++) {
					if (i == j)
						aRow[j] = betas.get(U.get(i))[1] * alpha.get(U.get(j)) - 1;
					else
						aRow[j] = betas.get(U.get(i))[1] * alpha.get(U.get(j));
				}
				double yi = -betas.get(U.get(i))[0] - betas.get(U.get(i))[1] * b;
				y.add(yi);
			}
			
			List<Double> solution = solve(A, y); //solve Ax = y
			if (solution != null) {
				for (int j = 0; j < U.size(); j++) {
					int k = U.get(j);
					xStatistic[k] = solution.get(j);
				}
			}
			else {
				logger.info("Cannot estimate statistic for X by expectation (#estimateInverse), stop estimating for this statistic here because use of other method is wrong.");
				return null;
			}
		}
		
		//Estimating missing zi (zStatistic) by equation 4, based on current parameter.
		zStatistic = 0;
		for (int j = 0; j < xStatistic.length; j++) {
			zStatistic += alpha.get(j) * xStatistic[j];
		}
		
		//Balance process is removed because it is over-fitting or not stable. Balance process is the best in some cases. So list U is not used.
		return new Statistics(zStatistic, xStatistic);
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		int N = this.data.getZData().size();
		int n = this.data.getXData().get(0).length;
		
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
			double[] zVector = this.data.getZData().get(i);
			if (!Util.isUsed(zVector[1]))
				continue;
			
			double[] xVector = this.data.getXData().get(i);
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
	protected boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter, Object... info) {
		// TODO Auto-generated method stub
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		
		return ((ExchangedParameter)estimatedParameter).terminatedCondition(
				threshold, 
				(ExchangedParameter)currentParameter, 
				(ExchangedParameter)previousParameter);
	}

	
	/**
	 * Getting exchanged parameter. Actually, this method calls {@link #getParameter()}.
	 * @return exchanged parameter.
	 */
	public ExchangedParameter getExchangedParameter() {
		return (ExchangedParameter)getParameter();
	}
	
	
	/**
	 * Getting large statistics. Actually, this method calls {@link #getStatistics()}.
	 * @return large statistics.
	 */
	public LargeStatistics getLargeStatistics() {
		return (LargeStatistics)getStatistics();
	}
	
	
	/**
	 * This method can be used to estimate Z value with incomplete profile.
	 * In other words, it is possible to test with incomplete testing data.
	 */
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		ExchangedParameter parameter = this.getExchangedParameter(); 
		if (parameter == null || input == null)
			return null;
		List<Double> alpha = parameter.getAlpha();
		if (alpha == null || alpha.size() == 0)
			return null;
		
		Profile profile = null;
		if (input instanceof Profile)
			profile = (Profile)input;
		else
			profile = createProfile(this.attList, input);
		if (profile == null)
			return null;
		
		double[] xStatistic = new double[this.xIndices.size()];
		xStatistic[0] = 1;
		for (int j = 1; j < this.xIndices.size(); j++) {
			double xValue = extractRegressor(profile, j);
			if (Util.isUsed(xValue))
				xStatistic[j] = (double)transformRegressor(xValue, false);
			else
				xStatistic[j] = Constants.UNUSED;
		}
		
		Statistics stat = estimate(new Statistics(Constants.UNUSED, xStatistic), alpha, parameter.getBetas());
		if (stat == null)
			return null;
		else
			return transformResponse(stat.getZStatistic(), true);
	}
	
	
	/**
	 * Executing this algorithm by arbitrary input parameter.
	 * @param input arbitrary input parameter.
	 * @return result of execution. Return null if execution is failed.
	 */
	public Object executeIntel(Object...input) {
		return execute(input);
	}
	
	
	/**
	 * Getting internal data. Actually, this method returns the current statistics.
	 * @return internal data which is the current statistics.
	 */
	protected synchronized LargeStatistics getData() {
		return this.data;
	}
	
	
	/**
	 * Testing whether missing values are fulfilled.
	 * @return true if missing values are fulfilled.
	 */
	public synchronized boolean isMissingDataFilled() {
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
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		config.put(R_CALC_VARIANCE_FIELD, R_CALC_VARIANCE_DEFAULT); //This attribute is used for testing
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.getParameter() == null)
			return "";
		ExchangedParameter exParameter = ((ExchangedParameter)this.getParameter());
		List<Double> alpha = exParameter.getAlpha();
		if (alpha == null || alpha.size() == 0)
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
		
		buffer.append(": ");
		buffer.append("t=" + getCurrentIteration());
		buffer.append(", coeff=" + MathUtil.format(exParameter.getCoeff()));
		buffer.append(", z-variance=" + MathUtil.format(exParameter.getZVariance()));

		return buffer.toString();
	}


	@Override
	public String parameterToShownText(Object parameter, Object...info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof ExchangedParameter))
			return "";
		
		ExchangedParameter exParameter = ((ExchangedParameter)parameter);
		return exParameter.toString();
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
	 * @param input specified input. It is often profile.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in internal indices.
	 * So index 0 always indicates to value 1. 
	 * @return value of regressor (X) extracted from specified profile.
	 */
	protected double extractRegressor(Object input, int index) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return defaultExtractVariable(input, null, xIndices, index);
		else
			return defaultExtractVariable(input, attList, xIndices, index);
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
	public synchronized Object extractResponse(Object input) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return defaultExtractVariable(input, null, zIndices, 1);
		else
			return defaultExtractVariable(input, attList, zIndices, 1);
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
	 * Calculating variance from specified sample.
	 * @param inputSample specified sample.
	 * @return variance from specified sample.
	 */
	public synchronized double variance(Fetcher<Profile> inputSample) {
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
	 * @return estimated sample.
	 */
	public Fetcher<Profile> fulfill(Fetcher<Profile> inputSample) {
		// TODO Auto-generated method stub
		if (this.getParameter() == null)
			return null;
		
		RegressionEMImpl em = (RegressionEMImpl)this.newInstance();
		LargeStatistics stat = null;
		try {
			if (em.prepareInternalData(inputSample))
				stat = (LargeStatistics) this.expectation(this.getParameter(), em.getData());
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			stat = null;
		}
		if (stat == null)
			return null;
		
		int N = stat.getZData().size();
		AttributeList attRef = getSampleAttributeList(inputSample);
		List<Profile> profiles = Util.newList();
		for (int i = 0; i < N; i++) {
			Profile profile = new Profile(attRef);
			double[] xvector = stat.getXData().get(i);
			double z = stat.getZData().get(i)[1];
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
