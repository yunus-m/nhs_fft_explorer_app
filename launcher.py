#%%
import docker
import webbrowser

import docker.errors

client = docker.from_env()

tar_file = './fft_explorer_image.tar'
fft_image_tag = 'fft_explorer_image:latest'

#
# Load image in if not already
#
existing_image_tags = [tag for image in client.images.list() for tag in image.tags]
if fft_image_tag not in existing_image_tags:
    print('No existing image named fft_explorer_image:latest. Loading image...')
    with open(tar_file, 'rb') as f:
        client.images.load(f.read())
else:
    print('Existing image found. Not loading image.')

#
# Remove existing container [linked to fft image] if present
#
for container in client.containers.list():
    if container.image.tags[0] != fft_image_tag:
        continue
    print('Found exisiting container, removing...')

    try:
        # container.stop()
        # container.wait()
        container.kill()
        container.wait()
    except Exception as e:
        pass

    break

print('Starting new container...')
container = client.containers.run(
    fft_image_tag,
    # name=fft_container_name,
    ports={'8501/tcp': 8501},
    detach=True,
    remove=True
)

if container.status == 'running':
    print('Container running. Opening browser...')

    try:
        # webbrowser.open('http://localhost:8501', new=2)
        import os
        os.system('explorer.exe "http://localhost:8501"')
    except Exception as e:
        print(f'Failed to open browser: {e}')
else:
    print('Container failed to start running')