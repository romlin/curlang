import io
import json
import os
import tarfile
import tempfile
import zstandard as zstd

from tqdm.auto import tqdm

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class PackageManager:
    @staticmethod
    def _validate_curlang_path(path, is_input=True, operation=None):
        absolute_path = os.path.abspath(path)

        if is_input:
            if not os.path.exists(absolute_path):
                raise FileNotFoundError(
                    f"The path '{absolute_path}' does not exist."
                )

            if operation == 'pack':
                if not os.path.isdir(absolute_path):
                    raise ValueError(
                        f"Input for pack operation must be a directory. Got: {absolute_path}"
                    )

                toml_path = os.path.join(absolute_path, "curlang.toml")

                if not os.path.exists(toml_path):
                    raise ValueError(
                        f"Directory '{absolute_path}' is not a valid curlang. Missing curlang.toml file."
                    )

            if operation in ('unpack', 'sign'):
                if not os.path.isfile(absolute_path):
                    raise ValueError(
                        f"Input for {operation} operation must be a .curlang file."
                    )

                if not path.endswith('.curlang'):
                    raise ValueError(
                        f"Input must be .curlang file for {operation} operation"
                    )

        return absolute_path

    def pack(self, input_path, overwrite=False, prebuild=False, metadata=None):
        try:
            abs_input_path = self._validate_curlang_path(
                input_path,
                is_input=True,
                operation='pack'
            )

            output_path = abs_input_path + '.curlang'

            if os.path.exists(output_path):
                if overwrite:
                    try:
                        os.remove(output_path)
                    except Exception as e:
                        print(
                            f"Failed to remove existing file: {output_path}. Error: {str(e)}"
                        )
                        raise
                else:
                    raise FileExistsError(
                        f"The package '{output_path}' already exists."
                    )

            if os.path.isfile(abs_input_path):
                print("Input is a file. Compressing single file.")

                try:
                    with open(abs_input_path, 'rb') as f:
                        data = f.read()

                    filesize = len(data)
                    compression_level = 5

                    print(f"Compressing file ({filesize / 1024:.1f} KB)...")
                    compressed_data = zstd.compress(data, compression_level)
                    compressed_size = len(compressed_data)

                    print(
                        f"Compressed size: {compressed_size / 1024:.1f} KB ({compressed_size / filesize * 100:.1f}%)"
                    )

                    with open(output_path, 'wb') as f:
                        f.write(compressed_data)
                except KeyboardInterrupt:
                    print("\nOperation interrupted by user. Cleaning up...")

                    if os.path.exists(output_path):
                        os.remove(output_path)
                    raise
            elif os.path.isdir(abs_input_path):
                files_to_archive = []

                for root, dirs, files in os.walk(
                        abs_input_path,
                        followlinks=False
                ):
                    if '.git' in dirs:
                        dirs.remove('.git')

                    if not prebuild:
                        if 'build' in dirs:
                            dirs.remove('build')
                        if 'web' in dirs:
                            dirs.remove('web')

                    for d in dirs:
                        d_path = os.path.join(root, d)

                        if os.path.islink(d_path):
                            files_to_archive.append(
                                (
                                    d_path,
                                    os.path.relpath(
                                        d_path,
                                        start=abs_input_path
                                    )
                                )
                            )

                    for file in files:
                        file_path = os.path.join(root, file)

                        if not prebuild and (
                                '/build/' in file_path or
                                '/web/' in file_path or
                                file_path.endswith('/build') or
                                file_path.endswith('/web')
                        ):
                            continue

                        files_to_archive.append(
                            (
                                file_path,
                                os.path.relpath(
                                    file_path,
                                    start=abs_input_path
                                )
                            )
                        )
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_tar_path = os.path.join(temp_dir, "temp.tar")

                        with tarfile.open(temp_tar_path, 'w') as tar:
                            if metadata:
                                metadata_bytes = json.dumps(metadata).encode(
                                    'utf-8'
                                )
                                metadata_info = tarfile.TarInfo(
                                    'curlang-metadata.json'
                                )
                                metadata_info.size = len(metadata_bytes)
                                tar.addfile(
                                    metadata_info,
                                    io.BytesIO(metadata_bytes)
                                )

                            with tqdm(
                                    total=len(files_to_archive),
                                    desc="Adding files to archive",
                                    unit='file'
                            ) as pbar:
                                for file_path, arcname in files_to_archive:
                                    tar.add(
                                        file_path,
                                        arcname=arcname,
                                        recursive=False
                                    )
                                    pbar.update(1)

                        with open(temp_tar_path, 'rb') as f:
                            data = f.read()

                        filesize = len(data)
                        compression_level = 5

                        print(
                            f"Compressing archive ({filesize / 1024:.1f} KB)..."
                        )

                        compressed_data = zstd.compress(
                            data,
                            compression_level
                        )

                        compressed_size = len(compressed_data)

                        print(
                            f"Compressed size: {compressed_size / 1024:.1f} KB ({compressed_size / filesize * 100:.1f}%)"
                        )

                        with open(output_path, 'wb') as f:
                            f.write(compressed_data)
                except KeyboardInterrupt:
                    print("\nOperation interrupted by user. Cleaning up...")

                    if os.path.exists(output_path):
                        os.remove(output_path)
                    raise
            return output_path
        except Exception as e:
            raise

    def unpack(self, input_path, output_path=None):
        abs_input_path = self._validate_curlang_path(
            input_path,
            is_input=True,
            operation='unpack'
        )

        metadata = self.get_metadata(abs_input_path)

        if metadata and metadata.get('prebuilt'):
            print("Unpacking prebuilt package...")
        else:
            print("Unpacking package...")

        final_output_path = output_path if output_path else \
            os.path.splitext(abs_input_path)[0]

        os.makedirs(final_output_path, exist_ok=True)

        try:
            compressed_size = os.path.getsize(abs_input_path)
            chunk_size = 1024 * 1024 * 4

            with open(abs_input_path, 'rb') as compressed_file:
                decompressor = zstd.ZstdDecompressor()
                reader = decompressor.stream_reader(compressed_file)

                with tempfile.NamedTemporaryFile() as tmp_file:
                    with tqdm(
                            total=compressed_size,
                            desc="Decompressing",
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024
                    ) as pbar:
                        last_position = 0

                        while True:
                            chunk = reader.read(chunk_size)

                            if not chunk:
                                break

                            tmp_file.write(chunk)
                            current_position = compressed_file.tell()
                            pbar.update(current_position - last_position)
                            last_position = current_position

                        pbar.update(compressed_size - last_position)

                    tmp_file.seek(0)

                    try:
                        with tarfile.open(fileobj=tmp_file, mode='r:') as tar:
                            members = tar.getmembers()

                            with tqdm(
                                    members,
                                    desc="Extracting",
                                    unit='file',
                                    dynamic_ncols=True
                            ) as pbar:
                                for member in pbar:
                                    tar.extract(member, path=final_output_path)
                    except tarfile.ReadError:
                        tmp_file.seek(0)

                        with open(final_output_path, 'wb') as f:
                            f.write(tmp_file.read())

        except KeyboardInterrupt:
            print("\nOperation interrupted. Cleaning up...")

            if os.path.exists(final_output_path):
                shutil.rmtree(final_output_path)
            raise

        return final_output_path

    @staticmethod
    def get_metadata(input_path):
        try:
            sanitized_path = os.path.abspath(os.path.normpath(input_path))

            if not os.path.exists(sanitized_path):
                return None

            with open(sanitized_path, 'rb') as f:
                compressed_data = f.read()

            decompressed_data = zstd.decompress(compressed_data)

            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(decompressed_data)
                tmp_file.seek(0)

                with tarfile.open(fileobj=tmp_file, mode='r:') as tar:
                    try:
                        metadata_member = tar.getmember('curlang-metadata.json')
                        metadata_file = tar.extractfile(metadata_member)

                        if metadata_file:
                            return json.loads(
                                metadata_file.read().decode('utf-8')
                            )
                    except KeyError:
                        return None

        except Exception:
            return None

    def sign(
            self,
            input_path,
            output_path,
            private_key_path,
            hash_size=256,
            passphrase=None
    ):
        abs_input_path = self._validate_curlang_path(
            input_path,
            is_input=True,
            operation='sign'
        )

        abs_output_path = self._validate_curlang_path(
            output_path,
            is_input=False,
            operation='sign'
        )

        abs_key_path = self._validate_curlang_path(
            private_key_path,
            is_input=True
        )

        if hash_size not in [256, 384, 512]:
            raise ValueError(
                f"Invalid hash size {hash_size}. Must be 256, 384, or 512."
            )

        try:
            with open(abs_key_path, 'rb') as key_file:
                key_data = key_file.read()
        except PermissionError:
            raise PermissionError(
                f"Permission denied reading private key at {abs_key_path}"
            )

        try:
            private_key = serialization.load_pem_private_key(
                key_data,
                password=passphrase.encode(
                    'utf-8'
                ) if passphrase else None,
                backend=default_backend()
            )
        except TypeError:
            raise ValueError(
                "Private key is encrypted. Please provide the correct passphrase."
            )
        except ValueError:
            raise ValueError(
                "Invalid private key format or incorrect passphrase."
            )

        hash_algorithm = {
            256: hashes.SHA256(),
            384: hashes.SHA384(),
            512: hashes.SHA512()
        }[hash_size]

        try:
            with open(abs_input_path, 'rb') as f:
                data = f.read()
        except PermissionError:
            raise PermissionError(
                f"Permission denied reading package at {abs_input_path}"
            )

        try:
            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hash_algorithm),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hash_algorithm
            )
        except Exception as e:
            raise ValueError(f"Signature generation failed: {str(e)}")

        separator = b"---SIGNATURE_SEPARATOR---"
        combined_data = data + separator + signature

        try:
            with open(abs_output_path, 'wb') as f:
                f.write(combined_data)
        except PermissionError:
            raise PermissionError(
                f"Permission denied writing signed package to {abs_output_path}"
            )
