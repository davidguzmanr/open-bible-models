python evaluate-tts.py \
    --ground_truth_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-ewe-igbo-yoruba/ground-truth \
    --synthesized_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-ewe-igbo-yoruba/generated \
    --metadata_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-ewe-igbo-yoruba/test.csv \
    --output_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-ewe-igbo-yoruba/results.csv \
    --metrics utmos wer \
    --asr-lang yor_Latn \
    --system-name f5


python evaluate-tts.py \
    --ground_truth_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-ewe-igbo-yoruba-nt/ground-truth \
    --synthesized_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-ewe-igbo-yoruba-nt/generated \
    --metadata_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-ewe-igbo-yoruba-nt/test.csv \
    --output_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-ewe-igbo-yoruba-nt/results.csv \
    --metrics utmos wer \
    --asr-lang yor_Latn \
    --system-name f5

python evaluate-tts.py \
    --ground_truth_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-yoruba-nt/ground-truth \
    --synthesized_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-yoruba-nt/generated \
    --metadata_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-yoruba-nt/test.csv \
    --output_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/F5TTS_v1_Base_vocos_custom_open-bible-yoruba-nt/results.csv \
    --metrics utmos wer \
    --asr-lang yor_Latn \
    --system-name f5

python evaluate-tts.py \
    --ground_truth_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/open-bible-yoruba-nt-finetune/ground-truth \
    --synthesized_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/open-bible-yoruba-nt-finetune/generated \
    --metadata_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/open-bible-yoruba-nt-finetune/test.csv \
    --output_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS/synthesis_output/open-bible-yoruba-nt-finetune/results.csv \
    --metrics utmos wer \
    --asr-lang yor_Latn \
    --system-name f5


python evaluate-tts.py \
    --ground_truth_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/synthesis_output/yoruba-nt/ground-truth \
    --synthesized_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/synthesis_output/yoruba-nt/generated \
    --metadata_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/synthesis_output/yoruba-nt/test.csv \
    --output_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/synthesis_output/yoruba-nt/results.csv \
    --metrics utmos wer \
    --asr-lang yor_Latn \
    --system-name everyvoice

python evaluate-tts.py \
    --ground_truth_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/synthesis_output/yoruba-nt-finetuned-vocoder/ground-truth \
    --synthesized_dir /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/synthesis_output/yoruba-nt-finetuned-vocoder/generated \
    --metadata_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/synthesis_output/yoruba-nt-finetuned-vocoder/test.csv \
    --output_csv /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/synthesis_output/yoruba-nt-finetuned-vocoder/results.csv \
    --metrics utmos wer \
    --asr-lang yor_Latn \
    --system-name everyvoice