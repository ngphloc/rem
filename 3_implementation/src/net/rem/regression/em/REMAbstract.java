/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.em;

import static net.rem.regression.RMAbstract.solve;

import java.awt.Color;
import java.rmi.RemoteException;
import java.util.List;

import flanagan.math.Fmath;
import flanagan.plot.PlotGraph;
import net.hudup.core.Util;
import net.hudup.core.alg.MemoryBasedAlg;
import net.hudup.core.alg.MemoryBasedAlgRemote;
import net.hudup.core.alg.NoteAlg;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.xURI;
import net.rem.em.EM;
import net.rem.em.EMRemote;
import net.rem.em.ExponentialEM;
import net.rem.regression.LargeStatistics;
import net.rem.regression.RMAbstract;
import net.rem.regression.VarWrapper;
import net.rem.regression.ui.graph.Graph;
import net.rem.regression.ui.graph.PlotGraphExt;

/**
 * This class is abstract class for EM algorithm {@link EM}.
 * 
 * @author Loc Nguyen
 * @version 2.0
 *
 */
public abstract class REMAbstract extends ExponentialEM implements REM, REMRemote, MemoryBasedAlg, MemoryBasedAlgRemote, NoteAlg {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public REMAbstract() {
		super();
	}

	
	/**
	 * Initialization parameter without data.
	 * @param regressorNumber the number of regressors including x1, x2,..., xn.
	 * @param random if true, randomization is processed.
	 * @return exchanged parameter.
	 */
	protected abstract ExchangedParameter initializeParameterWithoutData(int regressorNumber, boolean random);

	
	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {EMRemote.class.getName(), REMRemote.class.getName(), MemoryBasedAlgRemote.class.getName()};
	}


	/**
	 * Getting exchanged parameter. Actually, this method calls {@link #getParameter()} method.
	 * @return exchanged parameter.
	 */
	protected ExchangedParameter getExchangedParameter() {
		try {
			return (ExchangedParameter)getParameter();
		}
		catch (Throwable e) {
			LogUtil.trace(e);
		}
		
		return null;
	}
	
	
	@Override
	public LargeStatistics getLargeStatistics() throws RemoteException {
		return (LargeStatistics)getStatistics(); //The statistics is fulfilled data by expectation method.
	}
	
	
	@Override
	public synchronized String getDescription() throws RemoteException {
		if (this.getParameter() == null)
			return "";
		ExchangedParameter exParameter = ((ExchangedParameter)this.getParameter());
		List<Double> alpha = exParameter.getAlpha();
		if (alpha == null || alpha.size() == 0)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append(transformResponse(extractResponse().toString(), false) + " = " + MathUtil.format(alpha.get(0)));
		for (int j = 0; j < alpha.size() - 1; j++) {
			double coeff = alpha.get(j + 1);
			String regressorExpr = "(" + transformRegressor(extractRegressor(j + 1).toString(), false).toString() + ")";
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + regressorExpr);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + regressorExpr);
		}
		
		buffer.append(": ");
		buffer.append("t=" + getCurrentIteration());
		buffer.append(", coeff=" + MathUtil.format(exParameter.getCoeff()));
		buffer.append(", z-variance=" + MathUtil.format(exParameter.getZVariance()));

		if (exParameter.getXNormalDisParameter() != null)
			buffer.append(", x-parameter=(" + exParameter.getXNormalDisParameter().toString() + ")");

		return buffer.toString();
	}


	@Override
	public String parameterToShownText(Object parameter, Object...info) throws RemoteException {
		if (parameter == null || !(parameter instanceof ExchangedParameter))
			return "";
		
		ExchangedParameter exParameter = ((ExchangedParameter)parameter);
		return exParameter.toString();
	}

	
	@Override
	public synchronized Inspector getInspector() {
		return RMAbstract.getInspector(this);
	}


	/**
	 * Transforming independent variable X.
	 * In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param x specified variable X.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of X.
	 */
	protected Object transformRegressor(Object x, boolean inverse) {
		return x;
	}


	@Override
	public Object transformResponse(Object z, boolean inverse) throws RemoteException {
		return z;
	}


	@Override
	public String note() {
		return note;
	}


	@Override
    public synchronized Graph createRegressorGraph(VarWrapper regressor) throws RemoteException {
		if (getLargeStatistics() == null || getExchangedParameter() == null)
			return null;
    	
		ExchangedParameter parameter = getExchangedParameter();
		double coeff0 = parameter.getAlpha().get(0);
		double coeff1 = parameter.getAlpha().get(regressor.getIndex());
		if (coeff1 == 0) return null;
			
		LargeStatistics stats = getLargeStatistics();
    	int ncurves = 2;
    	int npoints = stats.size();
    	double[][] data = PlotGraph.data(ncurves, npoints);
    	
    	for(int i = 0; i < npoints; i++) {
            data[0][i] = stats.getXData().get(i)[regressor.getIndex()];
            data[1][i] = stats.getZData().get(i)[1];
        }
    	
    	data[2][0] = Fmath.minimum(data[0]);
    	data[3][0] = coeff0 + coeff1 * data[2][0];
    	data[2][1] = Fmath.maximum(data[0]);
    	data[3][1] = coeff0 + coeff1 * data[2][1];

    	PlotGraphExt pg = new PlotGraphExt(data);

    	pg.setGraphTitle("Regressor plot");
    	pg.setXaxisLegend(extractRegressor(regressor.getIndex()).toString());
    	pg.setYaxisLegend(extractResponse().toString());
    	int[] popt = {1, 0};
    	pg.setPoint(popt);
    	int[] lopt = {0, 3};
    	pg.setLine(lopt);

    	pg.setBackground(Color.WHITE);
        return pg;
    }

    
	@Override
    public synchronized Graph createResponseGraph() throws RemoteException {
		return RMAbstract.createResponseGraph(this, this.getLargeStatistics());
    }
    
    
    @Override
    public synchronized Graph createErrorGraph() throws RemoteException {
    	return RMAbstract.createErrorGraph(this, this.getLargeStatistics());
    }

    
    @Override
    public synchronized List<Graph> createResponseRalatedGraphs() throws RemoteException {
    	return RMAbstract.createResponseRalatedGraphs(this);
    }
    
    
    @Override
    public synchronized double calcVariance() throws RemoteException {
    	return RMAbstract.calcVariance(this, this.getLargeStatistics());
    }
    
    
    @Override
    public synchronized double calcR() throws RemoteException {
    	return RMAbstract.calcR(this, this.getLargeStatistics());
    }
    

    @Override
	public synchronized double calcR(double factor, int index) throws RemoteException {
    	return RMAbstract.calcR(this, this.getLargeStatistics(), factor, index);
	}

	
	@Override
    public synchronized double[] calcError() throws RemoteException {
    	return RMAbstract.calcError(this, this.getLargeStatistics());
    }


	@Override
	public boolean saveLargeStatistics(xURI uri, int decimal) throws RemoteException {
		return RMAbstract.saveLargeStatistics(this, this.getLargeStatistics(), uri, decimal);
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
	 * Getting complete data from specified data.
	 * @param data specified data which can have missing values.
	 * @return complete data from specified data.
	 */
	protected static LargeStatistics getCompleteData(LargeStatistics data) {
		int N = data.getZData().size();
		List<double[]> xStatistic = Util.newList();
		List<double[]> zStatistic = Util.newList();
		for (int i = 0; i < N; i++) {
			double[] zVector = data.getZData().get(i);
			if (!Util.isUsed(zVector[1]))
				continue;
			
			double[] xVector = data.getXData().get(i);
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
			return null;
		else
			return new LargeStatistics(xStatistic, zStatistic);
	}
	

}
