from trainer.pipelines.bilstm_cnn_pipeline import BiLSTMCNNPipeline


if __name__ == '__main__':
    file_paths = [
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv'
    ]
    pipeline = BiLSTMCNNPipeline(file_paths)
    pipeline.run()