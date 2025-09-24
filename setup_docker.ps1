# Define image name
$imageName = "lucasdino/seq_dec_proc:latest"

# Build image without cache
docker build --no-cache -t $imageName .

# Push to Docker Hub
docker push $imageName

# Remove local image
docker rmi $imageName

# Prune everything: containers, images, volumes, networks
docker system prune -a -f --volumes