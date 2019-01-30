package chapter2;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FilenameUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

public class DataUtilities {
  private static Logger log = LoggerFactory.getLogger(DataUtilities.class);

  /** Data URL for downloading */
  public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  /** Location to save and extract the training/testing data */
  public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

  protected static void downloadData() throws Exception {
    // Create directory if required
    File directory = new File(DATA_PATH);
    if (!directory.exists())
      directory.mkdir();

    // Download file:
    String archizePath = DATA_PATH + "/mnist_png.tar.gz";
    File archiveFile = new File(archizePath);
    String extractedPath = DATA_PATH + "mnist_png";
    File extractedFile = new File(extractedPath);

    if (!archiveFile.exists()) {
      log.info("Starting data download (15MB)...");
      getMnistPNG();
      //Extract tar.gz file to output directory
      DataUtilities.extractTarGz(archizePath, DATA_PATH);
    } else {
      //Assume if archive (.tar.gz) exists, then data has already been extracted
      log.info("Data (.tar.gz file) already exists at {}", archiveFile.getAbsolutePath());
      if (!extractedFile.exists()) {
        //Extract tar.gz file to output directory
        DataUtilities.extractTarGz(archizePath, DATA_PATH);
      } else {
        log.info("Data (extracted) already exists at {}", extractedFile.getAbsolutePath());
      }
    }
  }

  public static void getMnistPNG() throws IOException {
    String tmpDirStr = System.getProperty("java.io.tmpdir");
    String archizePath = DATA_PATH + "/mnist_png.tar.gz";

    if (tmpDirStr == null) {
      throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir");
    }

    File f = new File(archizePath);
    if (!f.exists()) {
      DataUtilities.downloadFile(DATA_URL, archizePath);
      log.info("Data downloaded to ", archizePath);
    } else {
      log.info("Using existing directory at ", f.getAbsolutePath());
    }
  }

  /**
   * Download a remote file if it doesn't exist.
   * @param remoteUrl URL of the remote file.
   * @param localPath Where to download the file.
   * @return True if and only if the file has been downloaded.
   * @throws Exception IO error.
   */
  public static boolean downloadFile(String remoteUrl, String localPath) throws IOException {
    boolean downloaded = false;
    if (remoteUrl == null || localPath == null)
      return downloaded;
    File file = new File(localPath);
    if (!file.exists()) {
      file.getParentFile().mkdirs();
      HttpClientBuilder builder = HttpClientBuilder.create();
      CloseableHttpClient client = builder.build();
      try (CloseableHttpResponse response = client.execute(new HttpGet(remoteUrl))) {
        HttpEntity entity = response.getEntity();
        if (entity != null) {
          try (FileOutputStream outstream = new FileOutputStream(file)) {
            entity.writeTo(outstream);
            outstream.flush();
          }
        }
      }
      downloaded = true;
    }
    if (!file.exists())
      throw new IOException("File doesn't exist: " + localPath);
    return downloaded;
  }

  /**
   * Extract a "tar.gz" file into a local folder.
   * @param inputPath Input file path.
   * @param outputPath Output directory path.
   * @throws IOException IO error.
   */
  public static void extractTarGz(String inputPath, String outputPath) throws IOException {
    if (inputPath == null || outputPath == null)
      return;
    final int bufferSize = 4096;
    if (!outputPath.endsWith("" + File.separatorChar))
      outputPath = outputPath + File.separatorChar;
    try (TarArchiveInputStream tais = new TarArchiveInputStream(
        new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(inputPath))))) {
      TarArchiveEntry entry;
      while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
        if (entry.isDirectory()) {
          new File(outputPath + entry.getName()).mkdirs();
        } else {
          int count;
          byte data[] = new byte[bufferSize];
          FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
          BufferedOutputStream dest = new BufferedOutputStream(fos, bufferSize);
          while ((count = tais.read(data, 0, bufferSize)) != -1) {
            dest.write(data, 0, count);
          }
          dest.close();
        }
      }
    }
  }

}