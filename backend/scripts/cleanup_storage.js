import mongoose from 'mongoose';
import dotenv from 'dotenv';
dotenv.config();

const uri = process.env.MONGODB_URI;
if (!uri) {
  console.error('No MONGODB_URI found');
  process.exit(1);
}

async function main() {
  await mongoose.connect(uri, { dbName: process.env.MONGODB_DB || 'arogyavaani' });
  const db = mongoose.connection.db;
  const bucket = new mongoose.mongo.GridFSBucket(db, { bucketName: 'model_artifacts' });

  // 1. Get all active global models
  const allModels = await db.collection('globalmodels').find({}).toArray();
  const activeModels = allModels.filter(m => m.status === 'ACTIVE' || m.status === 'BOOTSTRAP');
  const keepKeys = new Set(activeModels.map(m => m.artifactKey).filter(Boolean));

  console.log('Active models to protect:');
  for (const k of keepKeys) {
    console.log(' - ' + k);
  }

  // 2. Scan all files in GridFS
  const files = await db.collection('model_artifacts.files').find({}).toArray();
  console.log(`\nFound ${files.length} total files in GridFS:`);

  let toDelete = [];
  let toKeep = [];

  for (const f of files) {
    if (keepKeys.has(f.filename)) {
      toKeep.push(f);
    } else {
      toDelete.push(f);
    }
  }

  console.log(`\nFiles to keep (${toKeep.length}):`);
  for (const f of toKeep) {
    console.log(`  [KEEP] ${f.filename} (${(f.length / (1024 * 1024)).toFixed(2)} MB)`);
  }

  console.log(`\nFiles to prune/delete (${toDelete.length}):`);
  let deleteTotalBytes = 0;
  for (const f of toDelete) {
    console.log(`  [PRUNE] ${f.filename} (${(f.length / (1024 * 1024)).toFixed(2)} MB)`);
    deleteTotalBytes += f.length;
  }

  console.log(`\nTotal space that will be freed: ${(deleteTotalBytes / (1024 * 1024)).toFixed(2)} MB`);

  // Let's delete the orphaned and superseded files from GridFS
  for (const f of toDelete) {
    await bucket.delete(f._id);
    console.log(`Deleted: ${f.filename}`);
  }

  // Also remove superseded global model metadata docs if desired or keep metadata
  console.log('\nGridFS cleanup finished!');

  // Check stats after cleanup
  const chunksStats = await db.command({ collStats: 'model_artifacts.chunks' });
  const filesStats = await db.command({ collStats: 'model_artifacts.files' });
  console.log(`Updated model_artifacts.chunks logical size: ${(chunksStats.size / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`Updated model_artifacts.chunks count: ${chunksStats.count}`);
  console.log(`Updated model_artifacts.files count: ${filesStats.count}`);

  await mongoose.disconnect();
  process.exit(0);
}

main().catch(err => {
  console.error('Error during cleanup:', err);
  process.exit(1);
});
