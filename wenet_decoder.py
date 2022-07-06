import utils
import os, requests, tarfile, json, time
import wenet
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('wav_folder', type=str, help='The folder storing wav files.')
argparser.add_argument('--model_dir', type=str, default=None, help='The model saved directory.')
args = argparser.parse_args()

# download pre-trained model
url = r'https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/librispeech/20210610_u2pp_conformer_libtorch.tar.gz'
default_dir = os.path.join('cache', '20210610_u2pp_conformer_libtorch')
if args.model_dir is None:
    if os.path.exists(default_dir) is False:
        print('...Downloading pre-trained model from:', url)
        with open(utils.makedir(default_dir + '.tar.gz'), 'wb') as targz_f:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                targz_f.write(response.raw.read())
        print('...Extracting to:', default_dir)
        with tarfile.open(default_dir + '.tar.gz', 'r:gz') as targz_f:
            targz_f.extractall(os.path.dirname(default_dir))
            targz_f.close()
    args.model_dir = default_dir

# decode wav files
decoder = wenet.Decoder(lang='en', model_dir=args.model_dir)
for wav in utils.find_all_ext(args.wav_folder, 'wav'):
    s = time.time()
    ans = json.loads(decoder.decode_wav(wav))
    print(time.time() - s)
    decoder.reset()
    print(wav, ans['nbest'][0]['sentence'])

