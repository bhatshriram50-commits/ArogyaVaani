import fs from 'fs';
import crypto from 'crypto';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
dotenv.config();

async function main() {
  const filePath = 'model_artifacts/models/retinopathy/d4dcb89c-fef2-4ef8-9d6b-12997f2bbc23.avm1';
  const data = fs.readFileSync(filePath);
  const hash = crypto.createHash('sha256').update(data).digest('hex');
  const manifestLength = data.readUInt32LE(4);
  const manifest = JSON.parse(data.slice(8, 8 + manifestLength).toString('utf8'));
  const parameterShapes = Object.fromEntries(manifest.tensors.map(t => [t.name, t.shape]));
  
  console.log('Manifest classes:', manifest.classes);
  console.log('Checksum:', hash);
  console.log('Tensors count:', manifest.tensors.length);
  
  const uri = process.env.MONGODB_URI;
  await mongoose.connect(uri, { dbName: 'arogyavaani' });
  const db = mongoose.connection.db;
  const bucket = new mongoose.mongo.GridFSBucket(db, { bucketName: 'model_artifacts' });
  const filename = 'models/retinopathy/d4dcb89c-fef2-4ef8-9d6b-12997f2bbc23.avm1';
  
  // Clean existing if any
  const existingFiles = await bucket.find({ filename }).toArray();
  for (const f of existingFiles) await bucket.delete(f._id);
  
  console.log('Uploading Retinopathy v0 to GridFS...');
  const uploadStream = bucket.openUploadStream(filename, { chunkSizeBytes: 1048576 });
  await new Promise((resolve, reject) => {
    uploadStream.on('finish', resolve);
    uploadStream.on('error', reject);
    uploadStream.end(data);
  });
  console.log('Upload to GridFS complete!');
  
  // Register in globalmodels collection
  const existingModel = await db.collection('globalmodels').findOne({ task: 'RETINOPATHY', version: 0 });
  const doc = {
    task: 'RETINOPATHY',
    version: 0,
    round: 5,
    samples: 0,
    hospitals: 0,
    classes: manifest.classes,
    architecture: manifest.architecture,
    parameterShapes,
    artifactKey: filename,
    checksum: hash,
    status: 'BOOTSTRAP',
    createdAt: new Date(),
    updatedAt: new Date()
  };
  
  if (existingModel) {
    await db.collection('globalmodels').updateOne({ _id: existingModel._id }, { $set: doc });
    console.log('Updated existing GlobalModel document.');
  } else {
    await db.collection('globalmodels').insertOne(doc);
    console.log('Inserted new GlobalModel document.');
  }
  
  const stats = await db.stats();
  console.log('Current DB Logical Size:', (stats.dataSize / (1024*1024)).toFixed(2), 'MB');
  await mongoose.disconnect();
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
