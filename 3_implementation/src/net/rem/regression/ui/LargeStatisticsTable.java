/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.File;
import java.rmi.RemoteException;
import java.util.List;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;
import javax.swing.table.DefaultTableModel;

import flanagan.math.Fmath;
import flanagan.plot.PlotGraph;
import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.UriAssoc;
import net.hudup.core.logistic.xURI;
import net.hudup.core.logistic.ui.SortableTable;
import net.hudup.core.logistic.ui.SortableTableModel;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.phoebe.math.FlanaganStat;
import net.rem.regression.LargeStatistics;
import net.rem.regression.RM;
import net.rem.regression.VarWrapper;

/**
 * This is the Java table to show large statistics.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LargeStatisticsTable extends SortableTable implements MouseListener {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public LargeStatisticsTable() {
    	super(new LargeStatisticsTableModel());
    	addMouseListener(this);

	}
	
	
	/**
	 * Update this table with specified regression model.
	 * @param rm specified regression model.
	 * @throws RemoteException if any error raises.
	 */
	public void update(RM rm) throws RemoteException {
		LargeStatisticsTableModel model = (LargeStatisticsTableModel)getModel();
		model.update(rm);
		init();
	}
	
	
	/**
	 * Getting regression model.
	 * @return regression model.
	 */
	public RM getRM() {
		return ((LargeStatisticsTableModel)getModel()).getRM();
	}
	
	
	/**
	 * Showing histogram for selected column.
	 * @throws RemoteException if any error raises.
	 */
	private void hist() throws RemoteException {
		int selectedColumn = getSelectedColumn();
		if (selectedColumn < 1)
			return;
		
		RM rm = getRM();
		if (rm == null) return;
		LargeStatistics stats = rm.getLargeStatistics();
		if (stats == null || stats.size() == 0) return;
		boolean isRegressor = (selectedColumn < stats.getXData().get(0).length) ? true : false;
		
		double[] columnVector = null;
		if (isRegressor)
			columnVector = DSUtil.toDoubleArray(stats.getXColumnStatistic(selectedColumn));
		else
			columnVector = DSUtil.toDoubleArray(stats.getZStatistic());
		double bintWidth = (Fmath.maximum(columnVector) - Fmath.minimum(columnVector)) / FlanaganStat.DEFAULT_BIN_NUMBER;
		PlotGraph histGraph = FlanaganStat.histogramBinsPlot2(columnVector, bintWidth);
		histGraph.setBackground(Color.WHITE);

		VarWrapper var = null;
		if (isRegressor)
			var = rm.extractRegressor(selectedColumn);
		else
			var = rm.extractResponse();
		final JDialog dlg = new JDialog(UIUtil.getDialogForComponent(this), "Histogram of \"" + var.toString() + "\"", true);
		dlg.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		dlg.setSize(400, 300);
		dlg.setLocationRelativeTo(UIUtil.getDialogForComponent(this));
		dlg.setLayout(new BorderLayout());
		JPanel body = new JPanel(new BorderLayout());
		dlg.add(body, BorderLayout.CENTER);
		body.add(histGraph, BorderLayout.CENTER);
		JPanel footer = new JPanel();
		dlg.add(footer, BorderLayout.SOUTH);
		JButton btnClose = new JButton("Close");
		btnClose.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				dlg.dispose();
			}
		});
		footer.add(btnClose);
		dlg.setVisible(true);
	}
	
	
	/**
	 * Create context menu.
	 * @return context menu.
	 * @throws RemoteException if any error raises.
	 */
	private JPopupMenu createContextMenu() throws RemoteException {
		RM rm = getRM();
		if (rm == null || rm.getLargeStatistics() == null || rm.getLargeStatistics().size() == 0)
			return null;
		
		JPopupMenu contextMenu = new JPopupMenu();
		
		JMenuItem miHist = UIUtil.makeMenuItem(null, "Hist", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					try {
						hist();
					} catch (Exception ex) {ex.printStackTrace();}
				}
			});
		contextMenu.add(miHist);
		
		return contextMenu;
	}

	
	@Override
	public void mouseClicked(MouseEvent e) {
		try {
			if(SwingUtilities.isRightMouseButton(e) ) {
				JPopupMenu contextMenu = createContextMenu();
				if(contextMenu != null) 
					contextMenu.show((Component)e.getSource(), e.getX(), e.getY());
			}
			else if (e.getClickCount() >= 2){
				hist();
			}
		}
		catch (Exception ex) {ex.printStackTrace();}
	}


	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub
	}


	@Override
	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
	}


	@Override
	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub
	}


	@Override
	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub
	}


	/**
	 * Show large statistics of specified regression model.
	 * @param comp parent component.
	 * @param rm specified regression model.
	 * @param modal if true, showing modal dialog.
	 * @throws RemoteException if any error raises.
	 */
	public static void showDlg(final Component comp, final RM rm, boolean modal) throws RemoteException {
		if (rm == null || rm.getLargeStatistics() == null || rm.getLargeStatistics().size() == 0) {
			JOptionPane.showMessageDialog(
					comp, 
					"Null regression model or empty large statistics", 
					"Cannot show large statistics dialog", 
					JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		LargeStatisticsTable tblStats = new LargeStatisticsTable();
		tblStats.update(rm);
		
		final JDialog dlg = new JDialog(UIUtil.getDialogForComponent(comp), "Large statistics", modal);
		dlg.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		dlg.setSize(400, 300);
		dlg.setLocationRelativeTo(UIUtil.getDialogForComponent(comp));
		dlg.setLayout(new BorderLayout());
		JPanel body = new JPanel(new BorderLayout());
		dlg.add(body, BorderLayout.CENTER);
		body.add(new JScrollPane(tblStats), BorderLayout.CENTER);
		JPanel footer = new JPanel();
		dlg.add(footer, BorderLayout.SOUTH);
		JButton btnExport = new JButton("Export");
		btnExport.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				UriAssoc uriAssoc = Util.getFactory().createUriAssoc(xURI.create(new File(".")));
				xURI chooseUri = uriAssoc.chooseUri(dlg, false, new String[] {"csv"}, new String[] {"CSV file"}, null, "csv");
				if (chooseUri == null) {
					JOptionPane.showMessageDialog(
							dlg, 
							"Invalid URI", 
							"Large statistics not exported", 
							JOptionPane.INFORMATION_MESSAGE);
					return;
				}
				
				String decimalText = JOptionPane.showInputDialog(dlg, "Please enter decimal number", Constants.DECIMAL_PRECISION);
				int decimal = Constants.DECIMAL_PRECISION;
				try {
					decimal = Integer.parseInt(decimalText);
				}
				catch (Exception ex) {
					decimal = Constants.DECIMAL_PRECISION;
				}
				decimal = decimal <= 0 ? Constants.DECIMAL_PRECISION : decimal; 
				
				boolean result = true;
				try {
					result = rm.saveLargeStatistics(chooseUri, decimal);
				} catch (Exception ex) {ex.printStackTrace();}
				if (result)
					JOptionPane.showMessageDialog(
						dlg,
						"Large statistics exported successfully",
						"Exported successfully",
						JOptionPane.INFORMATION_MESSAGE);
				else
					JOptionPane.showMessageDialog(
						dlg,
						"Large statistics exported failed",
						"Exported failed",
						JOptionPane.ERROR_MESSAGE);
			}
		});
		footer.add(btnExport);
		JButton btnClose = new JButton("Close");
		btnClose.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				dlg.dispose();
			}
		});
		footer.add(btnClose);
		dlg.setVisible(true);
	}
}


/**
 * This is the Java table model for large statistics.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class LargeStatisticsTableModel extends SortableTableModel {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal regression model.
	 */
	protected RM rm = null;
	
	
	/**
	 * Default constructor.
	 */
	public LargeStatisticsTableModel() {
		super();
		// TODO Auto-generated constructor stub
	}


	/**
	 * Update this model with specified regression model.
	 * @param rm specified regression model.
	 * @throws RemoteException if any error raises.
	 */
	public void update(RM rm) throws RemoteException {
		this.rm = rm;
		
		if (rm == null) {
			this.setDataVector(Util.newVector(), Util.newVector());
			return;
		}
		LargeStatistics stats = rm.getLargeStatistics();
		List<VarWrapper> regressors = rm.extractRegressors();
		VarWrapper response = rm.extractResponse();
		if (stats == null || regressors.size() == 0 || response == null) {
			this.setDataVector(Util.newVector(), Util.newVector());
			return;
		}
		
		Vector<String> columns = Util.newVector(regressors.size() + 1);
		columns.add("No");
		for (VarWrapper regressor : regressors) {
			columns.add(regressor.toString());
		}
		columns.add(response.toString());
		
		Vector<Vector<Object>> data = Util.newVector(stats.size());
		for (int i = 0; i < stats.size(); i++) {
			double[] xVector = stats.getXData().get(i);
			double[] zVector = stats.getZData().get(i);
			Vector<Object> row = Util.newVector(xVector.length + 1);
			row.add(i + 1);
			for (int j = 1; j < xVector.length; j++) {
				row.add(MathUtil.format(xVector[j]));
			}
			row.add(MathUtil.format(zVector[1]));
			
			data.add(row);
		}
		
		this.setDataVector(data, columns);
	}
	
	
	/**
	 * Getting regression model.
	 * @return regression model.
	 */
	public RM getRM() {
		return rm;
	}
	
	
	@Override
	public boolean isCellEditable(int row, int column) {
		return false;
	}


//	@Override
//    public boolean isSortable(final int column) {
//        return true;
//    }
    
    
}



/**
 * This is the Java table to show large statistics.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
class LargeStatisticsTable2 extends JTable implements MouseListener {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public LargeStatisticsTable2() {
    	super(new LargeStatisticsTableModel2());
    	addMouseListener(this);

	}
	
	
	/**
	 * Update this table with specified regression model.
	 * @param rm specified regression model.
	 * @throws RemoteException if any error raises.
	 */
	public void update(RM rm) throws RemoteException {
		LargeStatisticsTableModel2 model = (LargeStatisticsTableModel2)getModel();
		model.update(rm);
	}
	
	
	/**
	 * Getting regression model.
	 * @return regression model.
	 */
	public RM getRM() {
		return ((LargeStatisticsTableModel2)getModel()).getRM();
	}
	
	
	/**
	 * Showing histogram for selected column.
	 * @throws RemoteException if any error raises.
	 */
	private void hist() throws RemoteException {
		int selectedColumn = getSelectedColumn();
		if (selectedColumn < 1)
			return;
		
		RM rm = getRM();
		if (rm == null) return;
		LargeStatistics stats = rm.getLargeStatistics();
		if (stats == null || stats.size() == 0) return;
		boolean isRegressor = (selectedColumn < stats.getXData().get(0).length) ? true : false;
		
		double[] columnVector = null;
		if (isRegressor)
			columnVector = DSUtil.toDoubleArray(stats.getXColumnStatistic(selectedColumn));
		else
			columnVector = DSUtil.toDoubleArray(stats.getZStatistic());
		double bintWidth = (Fmath.maximum(columnVector) - Fmath.minimum(columnVector)) / FlanaganStat.DEFAULT_BIN_NUMBER;
		PlotGraph histGraph = FlanaganStat.histogramBinsPlot2(columnVector, bintWidth);
		histGraph.setBackground(Color.WHITE);

		VarWrapper var = null;
		if (isRegressor)
			var = rm.extractRegressor(selectedColumn);
		else
			var = rm.extractResponse();
		final JDialog dlg = new JDialog(UIUtil.getDialogForComponent(this), "Histogram of \"" + var.toString() + "\"", true);
		dlg.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		dlg.setSize(400, 300);
		dlg.setLocationRelativeTo(UIUtil.getDialogForComponent(this));
		dlg.setLayout(new BorderLayout());
		JPanel body = new JPanel(new BorderLayout());
		dlg.add(body, BorderLayout.CENTER);
		body.add(histGraph, BorderLayout.CENTER);
		JPanel footer = new JPanel();
		dlg.add(footer, BorderLayout.SOUTH);
		JButton btnClose = new JButton("Close");
		btnClose.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				dlg.dispose();
			}
		});
		footer.add(btnClose);
		dlg.setVisible(true);
	}
	
	
	/**
	 * Create context menu.
	 * @return context menu.
	 * @throws RemoteException if any error raises.
	 */
	private JPopupMenu createContextMenu() throws RemoteException {
		RM rm = getRM();
		if (rm == null || rm.getLargeStatistics() == null || rm.getLargeStatistics().size() == 0)
			return null;
		
		JPopupMenu contextMenu = new JPopupMenu();
		
		JMenuItem miHist = UIUtil.makeMenuItem(null, "Hist", 
			new ActionListener() {
				
				public void actionPerformed(ActionEvent e) {
					try {
						hist();
					} catch (Exception ex) {ex.printStackTrace();}
				}
			});
		contextMenu.add(miHist);
		
		return contextMenu;
	}

	
	@Override
	public void mouseClicked(MouseEvent e) {
		try {
			if(SwingUtilities.isRightMouseButton(e) ) {
				JPopupMenu contextMenu = createContextMenu();
				if(contextMenu != null) 
					contextMenu.show((Component)e.getSource(), e.getX(), e.getY());
			}
			else if (e.getClickCount() >= 2){
				hist();
			}
		}
		catch (RemoteException ex) {ex.printStackTrace();}
	}


	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub
	}


	@Override
	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
	}


	@Override
	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub
	}


	@Override
	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub
	}


	/**
	 * Show large statistics of specified regression model.
	 * @param comp parent component.
	 * @param rm specified regression model.
	 * @param modal if true, showing modal dialog.
	 * @throws RemoteException if any error raises.
	 */
	public static void showDlg(final Component comp, final RM rm, boolean modal) throws RemoteException {
		if (rm == null || rm.getLargeStatistics() == null || rm.getLargeStatistics().size() == 0) {
			JOptionPane.showMessageDialog(
					comp, 
					"Null regression model or empty large statistics", 
					"Cannot show large statistics dialog", 
					JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		LargeStatisticsTable2 tblStats = new LargeStatisticsTable2();
		tblStats.update(rm);
		
		final JDialog dlg = new JDialog(UIUtil.getDialogForComponent(comp), "Large statistics", modal);
		dlg.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		dlg.setSize(400, 300);
		dlg.setLocationRelativeTo(UIUtil.getDialogForComponent(comp));
		dlg.setLayout(new BorderLayout());
		JPanel body = new JPanel(new BorderLayout());
		dlg.add(body, BorderLayout.CENTER);
		body.add(new JScrollPane(tblStats), BorderLayout.CENTER);
		JPanel footer = new JPanel();
		dlg.add(footer, BorderLayout.SOUTH);
		JButton btnExport = new JButton("Export");
		btnExport.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				UriAssoc uriAssoc = Util.getFactory().createUriAssoc(xURI.create(new File(".")));
				xURI chooseUri = uriAssoc.chooseUri(dlg, false, new String[] {"csv"}, new String[] {"CSV file"}, null, "csv");
				if (chooseUri == null) {
					JOptionPane.showMessageDialog(
							dlg, 
							"Invalid URI", 
							"Large statistics not exported", 
							JOptionPane.INFORMATION_MESSAGE);
					return;
				}
				
				String decimalText = JOptionPane.showInputDialog(dlg, "Please enter decimal number", Constants.DECIMAL_PRECISION);
				int decimal = Constants.DECIMAL_PRECISION;
				try {
					decimal = Integer.parseInt(decimalText);
				}
				catch (Exception ex) {
					decimal = Constants.DECIMAL_PRECISION;
				}
				decimal = decimal <= 0 ? Constants.DECIMAL_PRECISION : decimal; 
				
				boolean result = true;
				try {
					result = rm.saveLargeStatistics(chooseUri, decimal);
				}
				catch (Exception ex) {
					ex.printStackTrace(); result = false;
				}
				if (result)
					JOptionPane.showMessageDialog(
						dlg,
						"Large statistics exported successfully",
						"Exported successfully",
						JOptionPane.INFORMATION_MESSAGE);
				else
					JOptionPane.showMessageDialog(
						dlg,
						"Large statistics exported failed",
						"Exported failed",
						JOptionPane.ERROR_MESSAGE);
			}
		});
		footer.add(btnExport);
		JButton btnClose = new JButton("Close");
		btnClose.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				dlg.dispose();
			}
		});
		footer.add(btnClose);
		dlg.setVisible(true);
	}
}


/**
 * This is the Java table model for large statistics.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
class LargeStatisticsTableModel2 extends DefaultTableModel {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal regression model.
	 */
	protected RM rm = null;
	
	
	/**
	 * Default constructor.
	 */
	public LargeStatisticsTableModel2() {
		super();
		// TODO Auto-generated constructor stub
	}


	/**
	 * Update this model with specified regression model.
	 * @param rm specified regression model.
	 * @throws RemoteException if any error raises.
	 */
	public void update(RM rm) throws RemoteException {
		this.rm = rm;
		
		if (rm == null) {
			this.setDataVector(Util.newVector(), Util.newVector());
			return;
		}
		LargeStatistics stats = rm.getLargeStatistics();
		List<VarWrapper> regressors = rm.extractRegressors();
		VarWrapper response = rm.extractResponse();
		if (stats == null || regressors.size() == 0 || response == null) {
			this.setDataVector(Util.newVector(), Util.newVector());
			return;
		}
		
		Vector<String> columns = Util.newVector(regressors.size() + 1);
		columns.add("No");
		for (VarWrapper regressor : regressors) {
			columns.add(regressor.toString());
		}
		columns.add(response.toString());
		
		Vector<Vector<Object>> data = Util.newVector(stats.size());
		for (int i = 0; i < stats.size(); i++) {
			double[] xVector = stats.getXData().get(i);
			double[] zVector = stats.getZData().get(i);
			Vector<Object> row = Util.newVector(xVector.length + 1);
			row.add(i + 1);
			for (int j = 1; j < xVector.length; j++) {
				row.add(MathUtil.format(xVector[j]));
			}
			row.add(MathUtil.format(zVector[1]));
			
			data.add(row);
		}
		
		this.setDataVector(data, columns);
	}
	
	
	/**
	 * Getting regression model.
	 * @return regression model.
	 */
	public RM getRM() {
		return rm;
	}
	
	
	@Override
	public boolean isCellEditable(int row, int column) {
		return false;
	}


}

