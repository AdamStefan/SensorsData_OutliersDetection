require(xts)
library(AnomalyDetection)

 # ['Ext_Tem','Ext_Tem_units','Ext_Umi','Ext_Umi_units','Ext_Vvi','Ext_Vvi_units','Int_Pu1','Int_Pu1_units','Int_Pu2',
                # 'Int_Pu2_units','Int_Tem','Int_Tem_histe','Int_Tem_ref','Int_Tem_units','Int_Umi','Int_Umi_histe','Int_Umi_ref',
                # 'Int_Umi_units','dataId','dataTimeStamp','id','recordStamp']
				
				
 # ['Ext_Tem','Ext_Umi','Ext_Vvi','Int_Pu1','Int_Pu2',
                # 'Int_Tem','Int_Umi','dataTimeStamp']
				
loadData <-function(){
	df = read.csv("D:/Working Projects/SensorsData_OutliersDetection/OutlierDetection/sensorData.csv")
	df$dataTimeStamp <- strptime(df$dataTimeStamp,format = "%Y-%m-%d %H:%M:%S")
	return(df)
}
								
outlierDetection <- function(data){
	columns <- c('Ext_Tem','Ext_Umi','Ext_Vvi','Int_Pu1','Int_Pu2','Int_Tem','Int_Umi','dataTimeStamp')
	data <- data[columns]
	return (data)
	# timeSr = xts(data[columns],data$dataTimeStamp)
	
	# AnomalyDetectionTs(data[c('dataTimeStamp','Ext_Tem')], max_anoms=0.02, direction='both', plot=TRUE)
	# AnomalyDetectionTs(data[c('dataTimeStamp','Ext_Umi')], max_anoms=0.02, direction='both', plot=TRUE)
	# AnomalyDetectionTs(data[c('dataTimeStamp','Ext_Vvi')], max_anoms=0.02, direction='both', plot=TRUE)
	# AnomalyDetectionTs(data[c('dataTimeStamp','Int_Pu1')], max_anoms=0.02, direction='both', plot=TRUE)
	# AnomalyDetectionTs(data[c('dataTimeStamp','Int_Pu2')], max_anoms=0.02, direction='both', plot=TRUE)
	# AnomalyDetectionTs(data[c('dataTimeStamp','Int_Tem')], max_anoms=0.02, direction='both', plot=TRUE)
	# AnomalyDetectionTs(data[c('dataTimeStamp','Int_Umi')], max_anoms=0.02, direction='both', plot=TRUE)	
}


